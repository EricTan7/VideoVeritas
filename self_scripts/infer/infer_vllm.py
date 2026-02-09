import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import argparse
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import re
from swift.llm.dataset.dataset.data_utils import DATASET_FACTORY_TEST
from swift.qwen_vl_utils import process_vision_info
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
import numpy as np
import random
FPS = int(os.environ.get('FPS', 3.0))


SYSTEM_PROMPT = """You are an expert video analyst.
Please think about the question as if you were a human pondering deeply. It’s encouraged to include self-reflection or verification in the reasoning process. Then, give a final verdict within <answer> </answer> tags."""
USER_PROMPT = """Is this video real or fake?"""


class LLMPredict():
    def __init__(self, args):
        super().__init__()

        self.model = LLM(
            model=args.model_path,
            mm_encoder_tp_mode="data",
            max_model_len=65536,
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 1, "video": 1},
            tensor_parallel_size=args.tensor_parallel_size
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=4096
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.args = args

    def batch_predict(self, batch_item):
        vllm_batch_inputs = self.prepare_batch_inputs(batch_item)
        outputs = self.model.generate(vllm_batch_inputs, self.sampling_params)

        output_text = [out.outputs[0].text for out in outputs]
        return output_text

    def prepare_batch_inputs(self, batch_item):
        batch_messages = []
        vllm_batch_inputs = []
        for item in batch_item:
            if isinstance(item["videos"], list):
                image_path = item["videos"][0]
            else:
                image_path = item["videos"]

            if image_path.endswith(".mp4"):
                message_mm = {
                    "type": "video", 
                    "video": image_path,
                    "fps": FPS,
                }
            else:
                message_mm = {"type": "image", "image": image_path}

            message = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        message_mm,
                        {"type": "text", "text": USER_PROMPT}
                    ]
                }
            ]
            batch_messages.append(message)
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(message, image_patch_size=self.processor.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True)

            video_kwargs['do_resize'] = False

            mm_data = {}
            if images is not None:
                mm_data['image'] = images
            if videos is not None:
                mm_data['video'] = videos

            vllm_batch_inputs.append({
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs
            })

        return vllm_batch_inputs


def get_classification_metrics(pred, label, prob=None, num_failed=None, num_failed_oppo=0):
    result_dict = {}
    pred_real = np.count_nonzero(pred==0)
    pred_fake = np.count_nonzero(pred==1)
    acc = np.sum(pred == label) / len(label)
    precision_fake = precision_score(label, pred, pos_label=1)
    recall_fake = recall_score(label, pred, pos_label=1)
    precision_real = precision_score(label, pred, pos_label=0)
    recall_real = recall_score(label, pred, pos_label=0)
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
            fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            fnr = 1 - tpr
            err = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        except:
            auc, err = "null", "null"
        try:
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




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='path to dataset annotations file')
    parser.add_argument('--model_path', type=str,
                        default='',
                        help='path to the model')
    parser.add_argument('--save_root', type=str,
                        default='./result',
                        help='path to the model')
    parser.add_argument('--data_name', type=str,
                        default='',
                        help='path to the model')
    parser.add_argument('--tensor_parallel_size', type=int,
                        default=4, help='')
    args = parser.parse_args()

    data_path = DATASET_FACTORY_TEST[args.data_name]
    with open(data_path, 'r') as f:
        data = json.load(f)

    model = LLMPredict(args)

    all_outputs = []
    all_images = []
    all_labels = []
    results_all = {}
    idx = 0
    for i in tqdm(range(0, len(data), args.batch_size)):
        try:
            batch_images = data[i:i + args.batch_size]
        except:
            batch_images = data[i:]   # last batch

        batch_output_text = model.batch_predict(batch_images)
        all_outputs.extend(batch_output_text)
        all_images.extend(batch_images)
        all_labels.extend([item["label"] for item in batch_images])

    all_res = {}
    for item, out, label in zip(all_images, all_outputs, all_labels):
        if isinstance(item["videos"], list):
            video = item["videos"][0]
        else:
            video = item["videos"]
        all_res[video] = {"pred": out, "label": label}

    ## Save Outputs
    save_path = os.path.join(args.save_root, f"{args.data_name}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=4, ensure_ascii=False)

    ## Calculate Metrics
    pred_all,label_all = [], []
    num_failed = 0
    for pred, label in zip(all_outputs, all_labels):
        try:
            pattern_answer = r'<answer>\s*(.*?)\s*</answer>'
            pred = re.search(pattern_answer, pred)
            pred = pred.group(1).lower().strip()
        except:
            num_failed += 1
            continue
        real_in_pred = "real" in pred.lower()
        fake_in_pred = "fake" in pred.lower()

        if real_in_pred and not fake_in_pred:
            pred_lbl = 0 
        elif fake_in_pred and not real_in_pred:
            pred_lbl = 1
        else:
            num_failed += 1
            continue
        pred_all.append(pred_lbl)
        label_all.append(label)

    pred_all, label_all = np.array(pred_all), np.array(label_all)
    res = get_classification_metrics(pred_all, label_all, num_failed=num_failed)
    print(res)
    