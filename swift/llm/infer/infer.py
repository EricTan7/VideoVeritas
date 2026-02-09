# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from tqdm import tqdm

from swift.llm import InferArguments, InferRequest, SwiftPipeline, load_dataset, prepare_model_template, sample_dataset
from swift.plugin import InferStats, MeanMetric, compute_rouge_bleu
from swift.utils import JsonlWriter, get_dist_setting, get_logger, is_dist, is_master, read_from_jsonl
from ..dataset.loader import DatasetLoader
from .infer_engine import AdapterRequest, PtEngine
from .protocol import RequestConfig
from .utils import InferCliState, get_cached_dataset

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
import re
import torch
import time
import os


logger = get_logger()



def get_classification_metrics(pred, label, prob=None, num_failed=None, num_failed_oppo=None):
    result_dict = {}

    pred_real = np.count_nonzero(pred==0)
    pred_fake = np.count_nonzero(pred==1)
    acc = np.sum(pred == label) / len(label)
    precision_fake = precision_score(label, pred, pos_label=1)
    recall_fake = recall_score(label, pred, pos_label=1)
    precision_real = precision_score(label, pred, pos_label=0)
    recall_real = recall_score(label, pred, pos_label=0)
    print("="*100)
    print(pred_real, pred_fake, num_failed, num_failed_oppo, acc, precision_fake, recall_fake, precision_real, recall_real)
    try:
        f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake)
    except:
        f1_fake = "null"
    try:
        f1_real = 2 * precision_real * recall_real / (precision_real + recall_real)
    except:
        f1_real = "null"

    real_idx = label == 0
    fake_idx = label == 1
    real_pred = pred[real_idx]
    fake_pred = pred[fake_idx]
    acc_real = np.sum(real_pred == 0) / len(real_pred)
    acc_fake = np.sum(fake_pred == 1) / len(fake_pred)
    acc = np.sum(pred == label) / len(label)

    if prob is not None:
        try:
            # auc
            fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            # eer
            fnr = 1 - tpr
            err = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        except:
            auc, err = "null", "null"
        try:
            # ap
            ap = metrics.average_precision_score(label, prob)
        except:
            ap = "null"

    result_dict['acc'] = acc
    result_dict['recall_fake'] = recall_fake
    result_dict['recall_real'] = recall_real
    result_dict['f1_real'] = f1_real
    result_dict['f1_fake'] = f1_fake
    if prob is not None:
        result_dict['auc'] = auc

    result_dict['pred_real'] = pred_real
    result_dict['pred_fake'] = pred_fake
    result_dict['failed'] = num_failed 
    result_dict['failed_oppo'] = num_failed_oppo

    result_dict['acc_real'] = acc_real
    result_dict['acc_fake'] = acc_fake
    result_dict['precision_fake'] = precision_fake
    
    result_dict['precision_real'] = precision_real
    
    if prob is not None:
        result_dict['eer'] = err
        result_dict['ap'] = ap

    return result_dict



class SwiftInfer(SwiftPipeline):
    args_class = InferArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], InferArguments]] = None) -> None:
        from swift.llm import merge_lora
        super().__init__(args)
        args = self.args
        if args.merge_lora:
            merge_lora(args, device_map='cpu')
        self.infer_kwargs = {}
        if args.infer_backend == 'vllm' and args.adapters:
            self.infer_kwargs['adapter_request'] = AdapterRequest('_lora', args.adapters[0])

        if args.infer_backend == 'pt':
            model, self.template = prepare_model_template(args)
            self.infer_engine = PtEngine.from_model_template(model, self.template, max_batch_size=args.max_batch_size)
            self.infer_engine.reranker_use_activation = args.reranker_use_activation
            # logger.info(f'model: {self.infer_engine.model}')
        else:
            self.template = args.get_template(None)
            self.infer_engine = self.get_infer_engine(args, self.template)
        self.random_state = np.random.RandomState(args.data_seed)

    def __getattr__(self, key: str):
        try:
            return super().__getattr__(key)
        except AttributeError:
            if 'infer_engine' in self.__dict__:
                return getattr(self.infer_engine, key)
            raise

    @staticmethod
    def get_infer_engine(args: InferArguments, template=None, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
            'template': template,
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
        if infer_backend in {'pt', 'vllm'}:
            kwargs['reranker_use_activation'] = args.reranker_use_activation
        if infer_backend == 'pt':
            from .infer_engine import PtEngine
            infer_engine_cls = PtEngine
            kwargs.update(args.get_model_kwargs())
            if hasattr(args, 'max_batch_size'):
                kwargs.update({'max_batch_size': args.max_batch_size})
        elif infer_backend == 'vllm':
            from .infer_engine import VllmEngine
            infer_engine_cls = VllmEngine
            kwargs.update(args.get_vllm_engine_kwargs())
            seed = args.seed
            if is_dist():
                # Ensure that different data-parallel processes have different seeds.
                seed += get_dist_setting()[0] // args.vllm_tensor_parallel_size
                kwargs['distributed_executor_backend'] = 'external_launcher'
            kwargs['seed'] = seed
        elif infer_backend == 'sglang':
            from .infer_engine import SglangEngine
            infer_engine_cls = SglangEngine
            kwargs.update(args.get_sglang_engine_kwargs())
        else:
            from .infer_engine import LmdeployEngine
            infer_engine_cls = LmdeployEngine
            kwargs.update(args.get_lmdeploy_engine_kwargs())
        return infer_engine_cls(**kwargs)

    def run(self) -> List[Dict[str, Any]]:
        args = self.args

        file_name = args.result_path.split('/')[-1]
        result_path = os.path.join(os.path.dirname(args.result_path), f"{self.args.val_dataset[0]}-{file_name}")
        self.jsonl_writer = JsonlWriter(result_path) if result_path else None

        metric_path = os.path.join(os.path.dirname(args.result_path), "metrics.jsonl")
        self.jsonl_writer_metric = JsonlWriter(metric_path) if args.result_path else None
        if args.eval_human:
            result = self.infer_cli()
        else:
            result = self.infer_dataset()
        if args.result_path:
            logger.info(f'The inference results have been saved to result_path: `{args.result_path}`.')
        return result

    @staticmethod
    def parse_data_from_response(response):
        if hasattr(response, 'choices'):
            return response.choices[0].message.content
        elif hasattr(response, 'data'):
            emb = response.data[0].embedding
            shape = len(emb)
            sample = str(emb)
            if len(emb) > 6:
                sample = str(emb[:3])[:-1] + ', ..., ' + str(emb[-3:])[1:]
            return f'Embedding(shape: [1, {shape}]): {sample}'

    def infer_single(self, infer_request: Union[InferRequest, Dict[str, Any]], request_config: RequestConfig) -> str:
        res_or_gen = self.infer([infer_request],
                                request_config,
                                template=self.template,
                                use_tqdm=False,
                                **self.infer_kwargs)[0]
        if request_config and request_config.stream:
            response = ''
            for res in res_or_gen:
                delta = res.choices[0].delta.content
                print(delta, end='', flush=True)
                response += delta
            print()
        else:
            response = self.parse_data_from_response(res_or_gen)
            print(response)
        print('-' * 50)
        return response

    def infer_cli(self) -> List[Dict[str, Any]]:
        args = self.args
        template = self.template
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        support_multi_round = template.template_meta.support_multi_round
        if support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')

        infer_state = InferCliState()
        result_list = []
        while True:
            if not support_multi_round:
                infer_state.clear()
            query = infer_state.input_text()
            if query.strip().lower() in {'exit', 'quit'}:
                break
            query = infer_state.check_query(query)
            if query is None:
                continue
            infer_state.add_query(query)
            if args.model_meta.is_multimodal:
                infer_state.input_mm_data()
            if args.model_meta.is_reward or args.task_type == 'prm':
                # reward model
                response = infer_state.input_text()
                infer_state.add_response(response)
                data = infer_state.to_dict()
                response = self.infer_single(data, request_config)
                data = {'response': response, **data}
            else:
                data = infer_state.to_dict()
                response = self.infer_single(data, request_config)
                infer_state.add_response(response)
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, **data}
            result_list.append(data)
            if self.jsonl_writer:
                self.jsonl_writer.append(data)

        return result_list

    def _prepare_val_dataset(self) -> HfDataset:
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        if args.cached_dataset:
            _, val_datasets = get_cached_dataset(self.args)
        else:
            val_datasets = []
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            val_datasets.append(val_dataset)
        elif args.dataset:
            _, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
            val_datasets.append(val_dataset)
        assert len(val_datasets) > 0
        val_dataset = DatasetLoader._concat_datasets(val_datasets)
        val_dataset = sample_dataset(val_dataset, args.val_dataset_sample, args.dataset_shuffle, self.random_state)
        return val_dataset

    def _calc_metric(self):
        args = self.args
        if not is_master():
            return
        data_list = read_from_jsonl(self.jsonl_writer.fpath)
        preds, labels = [], []
        logprobs = []
        for data in data_list:
            preds.append(data['response'])
            labels.append(data['labels'])
            if args.logprobs:
                logprobs.append(data['logprobs']['content'])
            else:
                logprobs.append(0)

        if args.metric == 'acc':
            mean_metric = MeanMetric()
            for pred, label in zip(preds, labels):
                mean_metric.update(pred == label)
            res = {'acc': mean_metric.compute()['value']}
        elif args.metric in ['self_acc', 'self_acc_tags']:
            gt_real, gt_fake = ['real', 'authentic', 'genuine', 'camera', 'natural', '>real', ' real'], ['fake', 'synthetic', 'generated', 'machine', 'artificial', 'algorithmical', '>fake', ' fake']
            pred_all, label_all = [], []
            num_failed = 0
            num_failed_oppo = 0
            prob_all = []
            for pred, label, logprob in zip(preds, labels, logprobs):
                if args.metric == "self_acc_tags":
                    try:
                        pattern_answer = r'<answer>\s*(.*?)\s*</answer>'
                        pred = re.search(pattern_answer, pred)
                        pred = pred.group(1).lower().strip()
                    except:
                        num_failed += 1
                        continue
                real_in_pred = any(word in pred.lower() for word in gt_real)
                fake_in_pred = any(word in pred.lower() for word in gt_fake)
                label = 0 if "real" in label.lower() else 1
                # label = 0 if "a" in label.lower() else 1

                if real_in_pred and not fake_in_pred:
                    pred_lbl = 0 
                elif fake_in_pred and not real_in_pred:
                    pred_lbl = 1
                else:
                    num_failed += 1
                    continue
    
                ### prepare output probs
                if args.logprobs:
                    for item in logprob:
                        if item["token"].lower() == pred.lower() or item["token"].lower() == f">{pred.lower()}" or item["token"].lower() == f" {pred.lower()}":    # 定位预测token位置
                            top_logprob = item['top_logprobs']
                            prob = top_logprob[0]['logprob']
                            prob_oppo = None
                            for top_token in top_logprob[1:]:   # 从top2 token开始
                                if top_token['token'].lower() in gt_real and pred_lbl == 1:
                                    prob_oppo = top_token['logprob']
                                elif top_token['token'].lower() in gt_fake and pred_lbl == 0:
                                    prob_oppo = top_token['logprob']
                            break
                    if prob_oppo is None:
                        num_failed_oppo += 1
                        # 10个以内找不到oppo，则取最后一个token
                        prob_oppo = top_logprob[-1]['logprob']
                        # continue
                    # else:
                    if pred_lbl == 1:
                        prob_normalized = torch.tensor([prob_oppo, prob]).softmax(dim=0)[1]
                    else:
                        prob_normalized = torch.tensor([prob, prob_oppo]).softmax(dim=0)[1]
                    prob_all.append(prob_normalized.item())
                    
                pred_all.append(pred_lbl)
                label_all.append(label)

            pred_all, label_all = np.array(pred_all), np.array(label_all)
            if args.logprobs:
                prob_all = np.array(prob_all)
                res = get_classification_metrics(pred_all, label_all, prob_all, num_failed, num_failed_oppo=num_failed_oppo)
            else:
                res = get_classification_metrics(pred_all, label_all, num_failed=num_failed, num_failed_oppo=num_failed_oppo)
        elif args.metric in ['self_acc_explain', 'self_acc_flexible']:
            # 第一句话 This is a real image.
            gt_real, gt_fake = ['real', 'authentic', 'genuine', 'camera', 'natural', '>real', ' real'], ['fake', 'synthetic', 'generated', 'machine', 'artificial', 'algorithmical', '>fake', ' fake']
            pred_all, label_all = [], []
            num_failed = 0
            num_failed_oppo = 0
            prob_all = []
            for pred, label, logprob in zip(preds, labels, logprobs):
                if args.metric == "self_acc_explain":
                    pred = pred.split(".")[0].lower().strip()
                else:
                    pred = pred.split(".")[-1].lower().strip()
                real_in_pred = any(word in pred.lower() for word in gt_real)
                fake_in_pred = any(word in pred.lower() for word in gt_fake)
                label = 0 if "real" in label.lower() else 1

                if real_in_pred and not fake_in_pred:
                    pred_lbl = 0
                    pred = "real"
                elif fake_in_pred and not real_in_pred:
                    pred_lbl = 1
                    pred = "fake"
                else:
                    num_failed += 1
                    continue
    
                ### prepare output probs
                if args.logprobs:
                    for item in logprob:
                        if item["token"].lower() == pred.lower() or item["token"].lower() == f">{pred.lower()}" or item["token"].lower() == f" {pred.lower()}":    # 定位预测token位置
                            top_logprob = item['top_logprobs']
                            prob = top_logprob[0]['logprob']
                            prob_oppo = None
                            for top_token in top_logprob[1:]:   # 从top2 token开始
                                if top_token['token'].lower() in gt_real and pred_lbl == 1:
                                    prob_oppo = top_token['logprob']
                                elif top_token['token'].lower() in gt_fake and pred_lbl == 0:
                                    prob_oppo = top_token['logprob']
                            break
                    if prob_oppo is None:
                        num_failed_oppo += 1
                        # 10个以内找不到oppo，则取最后一个token
                        prob_oppo = top_logprob[-1]['logprob']
                        # continue
                    # else:
                    if pred_lbl == 1:
                        prob_normalized = torch.tensor([prob_oppo, prob]).softmax(dim=0)[1]
                    else:
                        prob_normalized = torch.tensor([prob, prob_oppo]).softmax(dim=0)[1]
                    prob_all.append(prob_normalized.item())
                    
                pred_all.append(pred_lbl)
                label_all.append(label)

            pred_all, label_all = np.array(pred_all), np.array(label_all)
            if args.logprobs:
                prob_all = np.array(prob_all)
                res = get_classification_metrics(pred_all, label_all, prob_all, num_failed, num_failed_oppo=num_failed_oppo)
            else:
                res = get_classification_metrics(pred_all, label_all, num_failed=num_failed, num_failed_oppo=num_failed_oppo)
        elif args.metric in ['self_acc_choice', 'self_acc_choice_tags']:
            pred_all, label_all = [], []
            num_failed = 0
            num_failed_oppo = 0
            prob_all = []
            for pred, label, logprob in zip(preds, labels, logprobs):
                if args.metric == 'self_acc_choice_tags':
                    try:
                        pattern_answer = r'<answer>\s*(.*?)\s*</answer>'
                        pred = re.search(pattern_answer, pred)
                        pred = pred.group(1).lower().strip()
                    except:
                        num_failed += 1
                        continue
                label = label.strip().replace(' ','').replace('_','').replace('.','').replace('\n','').lower()
                pred = pred.strip().replace(' ','').replace('_','').replace('.','').replace('\n','').lower()
                # label = 0 if "real" in label else 1
                label = 0 if "a" in label else 1
                if pred == "a":
                    pred_lbl = 0
                elif pred == "b":
                    pred_lbl = 1
                else:
                    num_failed += 1
                    continue

                ### prepare output probs
                if args.logprobs:
                    for item in logprob:
                        if item["token"].lower() == pred.lower() or item["token"].lower() == f">{pred.lower()}":    # 定位预测token位置
                            top_logprob = item['top_logprobs']
                            prob = top_logprob[0]['logprob']
                            prob_oppo = None
                            for top_token in top_logprob[1:]:   # 从top2 token开始
                                if top_token['token'].lower() in ["a", "b", ">a", ">b"]:
                                    prob_oppo = top_token['logprob']
                            break
                    if prob_oppo is None:
                        num_failed_oppo += 1
                        # 10个以内找不到oppo，则取最后一个token
                        prob_oppo = top_logprob[-1]['logprob']
                        # continue
                    # else:
                    if pred_lbl == 1:
                        prob_normalized = torch.tensor([prob_oppo, prob]).softmax(dim=0)[1]
                    else:
                        prob_normalized = torch.tensor([prob, prob_oppo]).softmax(dim=0)[1]
                    prob_all.append(prob_normalized.item())

                pred_all.append(pred_lbl)
                label_all.append(label)

            pred_all, label_all = np.array(pred_all), np.array(label_all)
            if args.logprobs:
                prob_all = np.array(prob_all)
                res = get_classification_metrics(pred_all, label_all, prob_all, num_failed, num_failed_oppo=num_failed_oppo)
            else:
                res = get_classification_metrics(pred_all, label_all, num_failed=num_failed, num_failed_oppo=num_failed_oppo)
        elif args.metric == 'rouge':
            res = compute_rouge_bleu(preds, labels)
        logger.info(res)
        self.jsonl_writer_metric.append({"dataset": self.args.val_dataset})
        self.jsonl_writer_metric.append(res)

    def infer_dataset(self) -> List[Dict[str, Any]]:
        args = self.args
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        val_dataset = self._prepare_val_dataset()
        logger.info(f'val_dataset: {val_dataset}')

        self.infer_kwargs['metrics'] = [InferStats()]
        if request_config and request_config.stream:
            result_list = []
            for data in val_dataset:
                labels = InferRequest.remove_response(data['messages'])
                query = data['messages'][-1]['content']
                print(f'[QUERY] {query}')
                if labels:
                    print(f'[LABELS] {labels}')
                print('[RESPONSE] ', end='')
                response = self.infer_single(data, request_config)
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, 'labels': labels, **data}
                result_list.append(data)
                if self.jsonl_writer:
                    self.jsonl_writer.append(data)
            metrics = self.infer_kwargs.pop('metrics')
            print(metrics[0].compute())
        else:
            if args.write_batch_size <= 0:
                args.write_batch_size = len(val_dataset)
            if args.write_batch_size < len(val_dataset) and args.result_path:
                logger.info(f'args.result_path: {args.result_path}')
            prog_bar = tqdm(
                total=len(val_dataset), dynamic_ncols=True, disable=args.write_batch_size >= len(val_dataset))
            result_list = []
            idx = 0
            while idx < len(val_dataset):
                shard_size = min(args.write_batch_size, len(val_dataset) - idx)
                shard_dataset = val_dataset.select(range(idx, idx + shard_size))
                result = self._batch_infer(shard_dataset, request_config)
                if self.jsonl_writer:
                    self.jsonl_writer.append(result, gather_obj=True)
                result_list += result
                idx += shard_size
                prog_bar.update(shard_size)
            prog_bar.close()
            metrics = self.infer_kwargs.pop('metrics')
            if result_list:
                metric = metrics[0].compute()
                print(f'[rank{args.rank}] {metric}' if args.rank >= 0 else str(metric))
        if args.metric is not None:
            self._calc_metric()
        return result_list

    def _batch_infer(self, val_dataset, request_config):
        args = self.args
        result_list = []
        if args.infer_backend == 'vllm':
            rank = args.rank // args.vllm_tensor_parallel_size if args.rank >= 0 else -1
            data_parallel_size = args.global_world_size // args.vllm_tensor_parallel_size
        else:
            rank, data_parallel_size = args.rank, args.global_world_size
        # The dataset is insufficient for DP partitioning
        if len(val_dataset) < data_parallel_size:
            if rank >= len(val_dataset):
                return []
            data_parallel_size = len(val_dataset)
        if rank >= 0 and data_parallel_size > 1:
            val_dataset = val_dataset.shard(data_parallel_size, rank, contiguous=True)
        val_dataset = list(val_dataset)
        labels_list = []
        for data in val_dataset:
            if args.task_type == 'causal_lm':
                labels = InferRequest.remove_response(data['messages'])
            else:
                labels = data.pop('label', None)
            labels_list.append(labels)

        resp_list = self.infer(val_dataset, request_config, template=self.template, use_tqdm=True, **self.infer_kwargs)
        if not (args.infer_backend == 'vllm' and rank >= 0
                and args.rank % args.vllm_tensor_parallel_size != 0):  # DP & TP
            for data, resp, labels in zip(val_dataset, resp_list, labels_list):
                response = resp.choices[0].message.content
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, 'labels': labels, 'logprobs': resp.choices[0].logprobs, **data}
                result_list.append(data)
        return result_list


def infer_main(args: Optional[Union[List[str], InferArguments]] = None):
    return SwiftInfer(args).main()
