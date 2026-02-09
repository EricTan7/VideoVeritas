# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils.versions import require_version
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import RunningMoments

from swift.llm import to_device
from swift.utils import get_logger
from ..mixin import DataLoaderMixin, SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights
    from torchvision.utils import save_image
except ImportError:
    # 错误处理
    raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights, save_image = None, None, None, None, None
import numpy as np
from swift.llm.template.template.self_aug import flow_guided_deformation, motion_perturbation, fade_perturbation
import functools
from swift.utils import get_env_args

del HFDPOTrainer.__init__
logger = get_logger()


def new_gather_function(tensor):
    tensor_list = gather_object([tensor])
    tensor_list = [t[None] if t.ndim == 0 else t for t in tensor_list]
    return torch.concat(to_device(tensor_list, tensor.device), dim=0)





class DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        if 'loss_weights' in DPOConfig.__dict__:
            # trl >= 0.20
            self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
            self.loss_weights = args.loss_weights
        else:
            self.loss_type = args.loss_type

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        for loss_type in loss_types:
            if (loss_type in ['hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'apo_zero', 'apo_down']
                    and args.label_smoothing > 0):
                warnings.warn(
                    f'You are using the {loss_type} loss type that does not support label smoothing. The '
                    '`label_smoothing` parameter will be ignored. '
                    'Set `label_smoothing` to `0.0` to remove this warning.',
                    UserWarning,
                )
            if loss_type == 'kto_pair':
                raise ValueError('Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.')

        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.model_adapter_name = None
        self.reference_free = args.reference_free
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

        if 'bco_pair' in loss_types:
            self.running = RunningMoments(self.accelerator)
        if self.args.ld_alpha is not None:
            require_version('trl>=0.18', '`ld_alpha` requires that "trl>=0.18".')
        if self.template.packing:
            self.accelerator.gather_for_metrics = new_gather_function

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_ref_model: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(batch)
        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        labels = batch.pop('labels', None)
        if self.is_encoder_decoder:
            batch['labels'] = labels
        text_position_ids = batch.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = batch.get('position_ids')
        outputs = model(**batch, use_cache=False)
        all_logits = outputs.logits

        if all_logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            all_logits = all_logits[:, -labels.shape[1]:]

        if not self.is_encoder_decoder and self.template.sequence_parallel_size == 1:
            # Shift so that tokens < n predict n
            labels = torch.roll(labels, shifts=-1, dims=1)
        per_token_logps, mean_all_logits, loss_mask = self.get_per_token_logps(
            all_logits, labels, label_pad_token_id=self.label_pad_token_id)
        origin_per_token_logps = per_token_logps

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if 'ipo' in loss_types:
            size_completion = loss_mask.sum(dim=-1)
            per_token_logps = per_token_logps / size_completion

        output = {}
        if self.template.padding_free:
            cu_seqlens = self.get_cu_seqlens(text_position_ids, batch.get('logits_to_keep'))
            num_examples = (cu_seqlens.shape[0] - 1) // 2
            all_logps = per_token_logps.new_zeros((num_examples * 2, ))
            completion_lengths = (cu_seqlens[1:] - cu_seqlens[:-1])
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper

            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                length = end - start
                public_length = public_lengths[i % num_examples]
                if self.args.ld_alpha is not None and not is_ref_model and length > public_length:
                    front_logps = per_token_logps[:, start:start + public_length].sum()
                    rear_logps = per_token_logps[:, start + public_length:end].sum()
                    all_logps[i] = front_logps + self.args.ld_alpha * rear_logps
                else:
                    all_logps[i] = per_token_logps[:, start:end].sum()
            num_tokens = cu_seqlens[num_examples]
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['mean_rejected_logits'] = mean_all_logits[:, num_tokens:][loss_mask[:, num_tokens:]].mean()
        else:
            num_examples = labels.shape[0] // 2
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:num_examples][loss_mask[:num_examples]].mean()
            if self.args.ld_alpha is not None and not is_ref_model:
                completion_lengths = loss_mask.sum(dim=1)

                chosen_lengths = completion_lengths[:num_examples]
                rejected_lengths = completion_lengths[num_examples:]
                public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
                public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

                seq_len = per_token_logps.size(1)
                text_position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

                ld_mask = text_position_ids < public_lengths.unsqueeze(1)
                mask = text_position_ids < completion_lengths.unsqueeze(1)

                front_mask = (ld_mask & mask).float()
                rear_mask = (~ld_mask & mask).float()
                front_logps = (per_token_logps * front_mask).sum(dim=1)
                rear_logps = (per_token_logps * rear_mask).sum(dim=1)

                all_logps = front_logps + self.args.ld_alpha * rear_logps
            else:
                all_logps = per_token_logps.sum(-1)
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:num_examples][loss_mask[:num_examples]].mean()
            output['mean_rejected_logits'] = mean_all_logits[num_examples:][loss_mask[num_examples:]].mean()
        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss
        return output

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)


            
class DPOTrainer_self(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        if 'loss_weights' in DPOConfig.__dict__:
            # trl >= 0.20
            self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
            self.loss_weights = args.loss_weights
        else:
            self.loss_type = args.loss_type

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        for loss_type in loss_types:
            if (loss_type in ['hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'apo_zero', 'apo_down']
                    and args.label_smoothing > 0):
                warnings.warn(
                    f'You are using the {loss_type} loss type that does not support label smoothing. The '
                    '`label_smoothing` parameter will be ignored. '
                    'Set `label_smoothing` to `0.0` to remove this warning.',
                    UserWarning,
                )
            if loss_type == 'kto_pair':
                raise ValueError('Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.')

        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.model_adapter_name = None
        self.reference_free = args.reference_free
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

        if 'bco_pair' in loss_types:
            self.running = RunningMoments(self.accelerator)
        if self.args.ld_alpha is not None:
            require_version('trl>=0.18', '`ld_alpha` requires that "trl>=0.18".')
        if self.template.packing:
            self.accelerator.gather_for_metrics = new_gather_function

        # local_weights = Raft_Small_Weights.DEFAULT
        # local_model = raft_small(weights=local_weights).to("cuda")
        # local_model.eval()
        # self.raft_weights = local_weights
        # self.raft_model = local_model
        # self.aug_type = get_env_args('AUG_TYPE', str, None)

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_ref_model: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()

        ## aug
        # if "aug" in batch:
        #     for idx in range(len(batch["aug"])):

        # num_half_samples = batch.shape[0] // 2
        # pos, neg = batch[:num_half_samples], batch[num_half_samples:]
        # # import ipdb
        # # ipdb.set_trace()
        # if pos["input_ids"] == neg["input_ids"] and pos["pixel_values"] == neg["pixel_values"]:    # TODO: 精度问题？

        #     num = np.random.randint(1, 4)
        #     if self_aug == "randomaug":
        #         self_aug = np.random.choice(["deform", "motion", "fade"])
        #         if self_aug == "deform":
        #             strength = float(np.random.randint(20, 51))     # 20
        #             region = np.random.choice(["motion"]*4+["static"]*2+["any"]*2)   # 2/3 motion
        #             aug_duration = np.random.randint(4, 9)      # 4
        #             aug_func = functools.partial(flow_guided_deformation, num_deformations=num, strength=strength, duration=aug_duration, region=region, flow_model_instance=self.raft_model, flow_weights_enum=self.raft_weights)
        #         elif self_aug == "motion":
        #             strength = np.round(np.random.uniform(0.1, 2.0), 1)     # 0.5
        #             region = np.random.choice(["motion"]*4+["static"]*2+["any"]*2)   # 2/3 motion
        #             aug_func = functools.partial(motion_perturbation, strength=strength, num_perturbations=num, flow_model_instance=self.raft_model, flow_weights_enum=self.raft_weights)
        #         elif self_aug == "fade":
        #             aug_duration = np.random.randint(4, 9)
        #             aug_func = functools.partial(fade_perturbation, num_events=num, patch_size_ratio_range=(0.1, 0.4), fade_duration=aug_duration)
        #         else:
        #             raise ValueError(self_aug)
        #     else:
        #         aug_func = None
        

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        # print(batch["input_ids"].shape, batch["labels"].shape, batch["attention_mask"].shape, batch["pixel_values"].shape)
        # print(batch["labels"])
        # torch.Size([2, 1813]) torch.Size([2, 1813]) torch.Size([2, 1813]) torch.Size([8, 3, 448, 448])
        # tensor([[  -100,   -100,   -100,  ...,     29, 151645,    198],
        # [  -100,   -100,   -100,  ...,   -100,   -100,   -100]],
        if use_logits_to_keep:
            self.prepare_logits_to_keep(batch)
        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        labels = batch.pop('labels', None)
        loss_scale = batch.pop("loss_scale", None)

        ## use loss_scale to construct two set of mask
        # per_token_logps: shouldn't be masked
        # sft mask: loss_mask
        # dpo mask：guided by loss_scale
        dpo_mask = None
        if loss_scale is not None:
            # nll_mask = loss_mask
            dpo_mask = loss_scale == 1.0

        if self.is_encoder_decoder:
            batch['labels'] = labels
        text_position_ids = batch.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = batch.get('position_ids')
        outputs = model(**batch, use_cache=False)
        all_logits = outputs.logits

        if all_logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            all_logits = all_logits[:, -labels.shape[1]:]

        if not self.is_encoder_decoder and self.template.sequence_parallel_size == 1:
            # Shift so that tokens < n predict n
            labels = torch.roll(labels, shifts=-1, dims=1)
            dpo_mask = torch.roll(dpo_mask, shifts=-1, dims=1)
        per_token_logps, mean_all_logits, loss_mask = self.get_per_token_logps(
            all_logits, labels, label_pad_token_id=self.label_pad_token_id)
        origin_per_token_logps = per_token_logps

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if 'ipo' in loss_types:
            size_completion = loss_mask.sum(dim=-1)
            per_token_logps = per_token_logps / size_completion

        output = {}
        if self.template.padding_free:
            cu_seqlens = self.get_cu_seqlens(text_position_ids, batch.get('logits_to_keep'))
            num_examples = (cu_seqlens.shape[0] - 1) // 2
            all_logps = per_token_logps.new_zeros((num_examples * 2, ))
            completion_lengths = (cu_seqlens[1:] - cu_seqlens[:-1])
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper

            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                length = end - start
                public_length = public_lengths[i % num_examples]
                if self.args.ld_alpha is not None and not is_ref_model and length > public_length:
                    front_logps = per_token_logps[:, start:start + public_length].sum()
                    rear_logps = per_token_logps[:, start + public_length:end].sum()
                    all_logps[i] = front_logps + self.args.ld_alpha * rear_logps
                else:
                    all_logps[i] = per_token_logps[:, start:end].sum()
            num_tokens = cu_seqlens[num_examples]
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['mean_rejected_logits'] = mean_all_logits[:, num_tokens:][loss_mask[:, num_tokens:]].mean()
        else:
            num_examples = labels.shape[0] // 2
            ## use_nll_loss
            use_nll_loss = batch.pop("use_nll_loss", None)
            nll_loss_scale = torch.tensor(use_nll_loss[:num_examples]).to(labels.device).unsqueeze(1) if use_nll_loss else None
            if not is_ref_model:
                if nll_loss_scale is not None:
                    loss_mask_nll = nll_loss_scale * loss_mask[:num_examples]
                    nll_loss = -origin_per_token_logps[:num_examples][loss_mask_nll.bool()].mean()
                    nll_loss[torch.isnan(nll_loss)] = 0.
                    output['nll_loss'] = nll_loss
                else:
                    output['nll_loss'] = -origin_per_token_logps[:num_examples][loss_mask[:num_examples]].mean()
            if self.args.ld_alpha is not None and not is_ref_model:
                completion_lengths = loss_mask.sum(dim=1)

                chosen_lengths = completion_lengths[:num_examples]
                rejected_lengths = completion_lengths[num_examples:]
                public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
                public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

                seq_len = per_token_logps.size(1)
                text_position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

                ld_mask = text_position_ids < public_lengths.unsqueeze(1)
                mask = text_position_ids < completion_lengths.unsqueeze(1)

                front_mask = (ld_mask & mask).float()
                rear_mask = (~ld_mask & mask).float()
                front_logps = (per_token_logps * front_mask).sum(dim=1)
                rear_logps = (per_token_logps * rear_mask).sum(dim=1)

                all_logps = front_logps + self.args.ld_alpha * rear_logps
            else:
                # all_logps = per_token_logps.sum(-1)
                all_logps = (per_token_logps * dpo_mask.float()).sum(-1)
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            # output['mean_chosen_logits'] = mean_all_logits[:num_examples][loss_mask[:num_examples]].mean()
            # output['mean_rejected_logits'] = mean_all_logits[num_examples:][loss_mask[num_examples:]].mean()
            output['mean_chosen_logits'] = mean_all_logits[:num_examples][dpo_mask[:num_examples]].mean()
            output['mean_rejected_logits'] = mean_all_logits[num_examples:][dpo_mask[num_examples:]].mean()
        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss
        return output

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):   # inputs: dict_keys(['input_ids', 'labels', 'attention_mask', 'pixel_values'])
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)
