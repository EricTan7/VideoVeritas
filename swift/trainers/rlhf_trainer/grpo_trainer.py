# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import concurrent.futures
import inspect
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set, Sized

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from packaging import version
from transformers import PreTrainedModel
from transformers.trainer import Trainer
from trl import GRPOTrainer as HFGRPOTrainer
from trl.models import prepare_deepspeed
from trl.trainer import grpo_trainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_trainer import RepeatSampler, nanmax, nanmin, nanstd
from trl.trainer.utils import selective_log_softmax

from swift.llm import RowPreprocessor, Template, to_device
from swift.llm.template.template_inputs import TemplateInputs
from swift.plugin import orms, rm_plugins, prms
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response,
                         seed_worker, unwrap_model_for_generation, get_env_args)
from ..mixin import SwiftMixin
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (_ForwardRedirection, compute_chord_loss, get_even_process_data, identity_data_collator,
                    load_pil_img, make_chord_sft_dataset, patch_profiling_context, patch_profiling_decorator,
                    patch_save_last_checkpoint, replace_assistant_response_with_ids)

try:
    from trl.trainer.utils import entropy_from_logits
except ImportError:
    from .utils import entropy_from_logits

from torch.utils.data import Sampler
from collections import defaultdict


del HFGRPOTrainer.__init__
del HFGRPOTrainer.log
grpo_trainer.seed_worker = seed_worker  # fix transformers 4.51.3

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab



### Conditional优势耦合策略 自定义
def default_task_coupling(
        task_data: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        默认的通用任务耦合策略
        
        Args:
            task_data: {
                'task_name_1': {
                    'indices': [idx0, idx1, ...],
                    'rewards': Tensor[K],
                    'rewards_per_func': Tensor[K, num_funcs],
                    'base_advantages': Tensor[K],
                },
                'task_name_2': {...},
                ...
            }
        
        Returns:
            adjusted_task_data: 同样的结构，但添加了 'adjusted_advantages' 字段
        """
        # 计算各任务的平均表现
        task_means = {
            name: data['rewards'].mean().item()
            for name, data in task_data.items()
        }
        
        # 找出表现最好和最差的任务
        if len(task_means) > 0:
            best_task = max(task_means, key=task_means.get)
            worst_task = min(task_means, key=task_means.get)
            best_mean = task_means[best_task]
            worst_mean = task_means[worst_task]
        else:
            best_task = worst_task = None
        
        # 调整策略：表现差的任务受到惩罚
        adjusted_data = {}
        for task_name, data in task_data.items():
            adjusted_advantages = data['base_advantages'].clone()
            
            # 如果是表现最差的任务，并且与最好任务差距大，则衰减
            if task_name == worst_task and best_task is not None:
                gap = best_mean - worst_mean
                if gap > 0.3:  # 差距阈值
                    decay_factor = 0.5 + 0.5 * (1 - min(gap, 1.0))
                    adjusted_advantages = adjusted_advantages * decay_factor
            
            adjusted_data[task_name] = {
                **data,
                'adjusted_advantages': adjusted_advantages
            }
        
        return adjusted_data


## AIGC任务约束感知任务：在前期集中优化AIGC任务，AIGC任务达标后，再兼顾感知任务
def gated_task_coupling(
    task_data: Dict[str, Dict[str, torch.Tensor]],
    threshold: float = 0.8,  # 检测任务奖励达到 0.6 才放开感知任务
    tau: float = 0.02,        # 平滑系数
    min_perception_weight: float = 0.1 # 给感知任务留一个保底权重，防止完全不更新
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    检测引导的感知任务加权策略 (Sigmoid Gating)
    当AIGC检测任务达到阈值后，再兼顾感知任务
    
    逻辑：
    1. 计算 'aigc' 任务当前的平均表现。
    2. 通过 Sigmoid 函数计算出一个门控权重 gate。
    3. 如果是 'perception' 任务，其优势值乘上这个 gate。
    4. 'aigc' 任务保持 full strength。
    """
    
    # 1. 获取 AIGC 检测任务的表现
    # 建议使用当前 batch 的平均奖励作为参考
    if 'aigc' in task_data:
        aigc_performance = task_data['aigc']['rewards'].mean().item()
    else:
        # 如果 batch 中没检测任务，默认不打折（或者根据业务逻辑定）
        aigc_performance = 1.0 

    # 2. 计算门控权重 (Gate)
    # 使用 Sigmoid 使得权重在 threshold 附近平滑过渡
    # 当 aigc_performance == threshold 时，gate 约为 0.5
    gate = torch.sigmoid(torch.tensor((aigc_performance - threshold) / tau)).item()
    
    # 为了防止感知任务梯度完全消失导致模型死掉，可以设一个下限
    effective_gate = max(gate, min_perception_weight)

    # 3. 调整优势值
    adjusted_data = {}
    for task_name, data in task_data.items():
        # 标准化 Advantage 是非常重要的一步，防止不同任务量级不一致
        # 这里假设 base_advantages 已经过初步处理，如果没有，建议在此处做个简单的归一化
        adv = data['base_advantages'].clone()
        
        # 应用门控
        if task_name != 'aigc':
            # 只对感知任务进行打折
            adjusted_advantages = adv * effective_gate
        else:
            # 检测任务（aigc）或其他任务保持 1.0 权重
            adjusted_advantages = adv

        adjusted_data[task_name] = {
            **data,
            'adjusted_advantages': adjusted_advantages
        }
        
    # 打印一下当前的门控状态，方便调试查看
    # print(f"[Standard Logic] AIGC Perf: {aigc_performance:.3f}, Perc Gate: {effective_gate:.3f}")
    
    return adjusted_data




## AIGC任务约束感知任务：在前期集中优化AIGC任务，AIGC任务达标后，再兼顾感知任务
def hard_gating_coupling(
    task_data: Dict[str, Dict[str, torch.Tensor]],
    threshold: float = 0.9,  # 检测任务奖励达到 0.6 才放开感知任务
    tau: float = 0.05,        # 平滑系数
    min_perception_weight: float = 0.1 # 给感知任务留一个保底权重，防止完全不更新
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    AIGC奖励达到阈值前，将感知任务完全gating掉
    """
    
    # 1. 获取 AIGC 检测任务的表现
    # 建议使用当前 batch 的平均奖励作为参考
    if 'aigc' in task_data:
        aigc_performance = task_data['aigc']['rewards'].mean().item()
    else:
        # 如果 batch 中没检测任务，默认不打折（或者根据业务逻辑定）
        aigc_performance = 1.0 

    # 2. 计算门控权重 (Gate)
    if aigc_performance < threshold:    # 小于阈值，直接gating掉
        gate = 0.0
    else:
        gate = torch.sigmoid(torch.tensor((aigc_performance - threshold) / tau)).item()     # 大于阈值，平滑系数

    # 3. 调整优势值
    adjusted_data = {}
    for task_name, data in task_data.items():
        # 标准化 Advantage 是非常重要的一步，防止不同任务量级不一致
        # 这里假设 base_advantages 已经过初步处理，如果没有，建议在此处做个简单的归一化
        adv = data['base_advantages'].clone()
        
        # 应用门控
        if task_name != 'aigc':
            # 只对感知任务进行打折
            adjusted_advantages = adv * gate
        else:
            # 检测任务（aigc）或其他任务保持 1.0 权重
            adjusted_advantages = adv

        adjusted_data[task_name] = {
            **data,
            'adjusted_advantages': adjusted_advantages
        }
        
    # print("&"*50, aigc_performance, gate, adjusted_data["perception"]['base_advantages'], adjusted_data["perception"]['adjusted_advantages'])
    return adjusted_data



def hard_gating_aigc_bonus_coupling(
    task_data: Dict[str, Dict[str, torch.Tensor]],
    threshold: float = 0.9,  # 检测任务奖励达到 0.6 才放开感知任务
    tau: float = 0.05,        # 平滑系数
    min_perception_weight: float = 0.1 # 给感知任务留一个保底权重，防止完全不更新
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    AIGC奖励达到阈值前，将感知任务完全gating掉；
    batch-level perception bonus：感知做得好，给aigc任务bonus
    """
    
    # 1. 获取 AIGC 检测任务的表现
    # 建议使用当前 batch 的平均奖励作为参考
    aigc_rewards = task_data['aigc']['rewards']
    aigc_performance = aigc_rewards.mean().item()
    perception_performance = task_data['perception']['rewards'].mean().item()

    # 2. 感知gating
    if aigc_performance < threshold:    # 小于阈值，直接gating掉
        gate = 0.0
    else:
        gate = torch.sigmoid(torch.tensor((aigc_performance - threshold) / tau)).item()     # 大于阈值，平滑系数

    # AIGC bonus
    ## 给rewards值的bonus：只给答对的bonus
    aigc_bonus = torch.clamp(torch.log(torch.tensor(perception_performance)+1), 0.0, 0.3).item()

    # 3. 调整优势值
    adjusted_data = {}
    for task_name, data in task_data.items():
        # 标准化 Advantage 是非常重要的一步，防止不同任务量级不一致
        # 这里假设 base_advantages 已经过初步处理，如果没有，建议在此处做个简单的归一化
        adv = data['base_advantages'].clone()
        
        # 应用门控
        if task_name != 'aigc':
            # 只对感知任务进行打折
            adjusted_advantages = adv * gate
        else:
            # aigc任务加上bonus
            adjusted_rewards = torch.where(aigc_rewards>=1.0, aigc_rewards+aigc_bonus, aigc_rewards)
            adjusted_advantages = (adjusted_rewards-adjusted_rewards.mean()) / (adjusted_rewards.std()+1e-4)
            # adjusted_advantages = adv
            # print("@"*50, perception_performance, aigc_bonus, aigc_rewards, adjusted_rewards)

        adjusted_data[task_name] = {
            **data,
            'adjusted_advantages': adjusted_advantages
        }
        
    # print("@"*50, perception_performance, aigc_bonus, adjusted_data["aigc"]['base_advantages'], adjusted_data["aigc"]['adjusted_advantages'])
    return adjusted_data




def aigc_bonus_coupling(
    task_data: Dict[str, Dict[str, torch.Tensor]],
    threshold: float = 0.9,  # 检测任务奖励达到 0.6 才放开感知任务
    tau: float = 0.05,        # 平滑系数
    min_perception_weight: float = 0.1 # 给感知任务留一个保底权重，防止完全不更新
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    AIGC奖励达到阈值前，将感知任务完全gating掉；
    batch-level perception bonus：感知做得好，给aigc任务bonus
    """
    
    # 1. 获取 AIGC 检测任务的表现
    # 建议使用当前 batch 的平均奖励作为参考
    aigc_rewards = task_data['aigc']['rewards']
    aigc_performance = aigc_rewards.mean().item()
    perception_performance = task_data['perception']['rewards'].mean().item()

    # AIGC bonus
    ## 给rewards值的bonus：只给答对的bonus
    aigc_bonus = torch.clamp(torch.log(torch.tensor(perception_performance)+1), 0.0, 0.3).item()

    # 3. 调整优势值
    adjusted_data = {}
    for task_name, data in task_data.items():
        # 标准化 Advantage 是非常重要的一步，防止不同任务量级不一致
        # 这里假设 base_advantages 已经过初步处理，如果没有，建议在此处做个简单的归一化
        adv = data['base_advantages'].clone()
        
        # 应用门控
        if task_name != 'aigc':
            adjusted_advantages = adv
        else:
            # aigc任务加上bonus
            adjusted_rewards = torch.where(aigc_rewards>=1.0, aigc_rewards+aigc_bonus, aigc_rewards)
            adjusted_advantages = (adjusted_rewards-adjusted_rewards.mean()) / (adjusted_rewards.std()+1e-4)
            # adjusted_advantages = adv
            # print("@"*50, perception_performance, aigc_bonus, aigc_rewards, adjusted_rewards)

        adjusted_data[task_name] = {
            **data,
            'adjusted_advantages': adjusted_advantages
        }
        
    # print("@"*50, perception_performance, aigc_bonus, adjusted_data["aigc"]['base_advantages'], adjusted_data["aigc"]['adjusted_advantages'])
    return adjusted_data


    
def is_main_process():
    return dist.get_rank() == 0 if dist.is_initialized() else True

### ======== 同视频, 多任务对齐采样
class MultiTaskRepeatSampler(Sampler):
    """
    多任务重复采样器（适配GRPO，支持任意数量任务）
    
    核心特性：
    1. 同一视频的所有任务总是在一起
    2. 符合TRL RepeatSampler的采样逻辑
    3. 自动适应任务数量
    4. 支持跨GPU一致性
    
    Args:
        data_source: 数据集（必须包含video_id和task字段）
        task_names: 任务名称列表，决定输出顺序
                   如果为None，则自动从数据集推断并按字母序排列
        mini_repeat_count: 每个样本重复次数（对应num_generations）
        batch_size: 每个batch的样本数（必须能被任务数整除）
        repeat_count: batch级别的重复次数
        shuffle: 是否打乱
        seed: 随机种子
        strict_validation: 严格验证模式，缺失任务时报错
    
    Example:
        假设有3个视频，每个视频3个任务，batch_size=6（2个视频），
        mini_repeat_count=2，repeat_count=2
        
        数据结构：
        idx 0: video_0, task_a
        idx 1: video_0, task_b
        idx 2: video_0, task_c
        idx 3: video_1, task_a
        idx 4: video_1, task_b
        idx 5: video_1, task_c
        ...
        
        输出索引序列（按task_names=['task_a', 'task_b', 'task_c']顺序）:
        # Batch 1 (video_0, video_1)
        [0,0, 1,1, 2,2,  3,3, 4,4, 5,5,  # repeat 1: 每个任务重复2次
         0,0, 1,1, 2,2,  3,3, 4,4, 5,5]  # repeat 2
    """
    
    def __init__(
        self,
        data_source: Sized,
        task_names: List[str] = None,
        mini_repeat_count: int = 1,
        batch_size: int = 2,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
        strict_validation: bool = False,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed
        self.strict_validation = strict_validation
        
        # 构建视频-任务映射
        self.video_task_map = self._build_video_task_map()
        
        # 确定任务列表
        if task_names is None:
            # 自动推断：从数据中提取所有任务名并排序
            all_tasks = set()
            for tasks_dict in self.video_task_map.values():
                all_tasks.update(tasks_dict.keys())
            self.task_names = sorted(list(all_tasks))
            print(f"📋 自动检测到的任务（按字母序）: {self.task_names}")
        else:
            self.task_names = task_names
        
        self.num_tasks = len(self.task_names)
        
        # 验证batch_size
        if batch_size % self.num_tasks != 0:
            raise ValueError(
                f"batch_size ({batch_size}) 必须能被任务数 ({self.num_tasks}) 整除"
            )
        
        # 计算每个batch包含多少个视频
        self.videos_per_batch = batch_size // self.num_tasks
        
        # 验证每个视频是否有所有任务
        self.valid_videos = self._validate_and_filter_videos()
        self.num_videos = len(self.valid_videos)
        
        # 初始化随机数生成器
        if shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)
        if is_main_process():
            self._print_summary()
    
    def _build_video_task_map(self) -> Dict[str, Dict[str, int]]:
        """
        构建 video_id -> {task_name: sample_idx} 的映射
        
        Returns:
            {
                'video_0': {'task_a': 0, 'task_b': 1, 'task_c': 2},
                'video_1': {'task_a': 3, 'task_b': 4, 'task_c': 5},
                ...
            }
        """
        video_task_map = defaultdict(dict)
        
        for idx in range(len(self.data_source)):
            sample = self.data_source[idx]
            video_id = sample['video_id']
            task = sample['task']
            video_task_map[video_id][task] = idx
        
        return dict(video_task_map)
    
    def _validate_and_filter_videos(self) -> List[str]:
        """
        验证每个视频是否包含所有任务，并返回有效视频列表
        
        Returns:
            有效视频的video_id列表
        """
        valid_videos = []
        incomplete_videos = []
        
        for video_id, tasks_dict in self.video_task_map.items():
            # 检查是否有所有任务
            missing_tasks = set(self.task_names) - set(tasks_dict.keys())
            
            if missing_tasks:
                incomplete_videos.append((video_id, missing_tasks))
            else:
                valid_videos.append(video_id)
        
        # 处理不完整的视频
        if incomplete_videos:
            error_msg = f"发现 {len(incomplete_videos)} 个视频缺失任务配对"
            
            if len(incomplete_videos) <= 5:
                details = "\n".join([
                    f"  - {vid}: 缺失 {missing}"
                    for vid, missing in incomplete_videos[:5]
                ])
                error_msg += f":\n{details}"
            
            if self.strict_validation:
                raise ValueError(error_msg)
            else:
                print(f"⚠️  {error_msg}，已忽略这些视频")
        
        if len(valid_videos) == 0:
            raise ValueError("没有找到包含所有任务的完整视频！")
        
        return valid_videos
    
    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"MultiTaskRepeatSampler 初始化完成")
        print(f"{'='*60}")
        print(f"  数据集大小:        {len(self.data_source)}")
        print(f"  任务列表:          {self.task_names}")
        print(f"  任务数量:          {self.num_tasks}")
        print(f"  有效视频数:        {self.num_videos}")
        print(f"  batch_size:        {self.batch_size}")
        print(f"    -> 每batch视频数: {self.videos_per_batch}")
        print(f"    -> 每视频任务数:  {self.num_tasks}")
        print(f"  mini_repeat_count: {self.mini_repeat_count}")
        print(f"  repeat_count:      {self.repeat_count}")
        print(f"  有效batch数:       {self.num_videos // self.videos_per_batch}")
        print(f"  总样本数:          {len(self)}")
        print(f"{'='*60}\n")
    
    def __iter__(self):
        """
        生成采样索引序列
        
        核心逻辑：
        1. 打乱视频顺序（只一次）
        2. 分组成batch
        3. 对每个batch:
           - 重复repeat_count次
           - 对每个视频:
             - 按task_names顺序输出每个任务
             - 每个任务重复mini_repeat_count次
        """
        # Step 1: 获取视频索引并打乱（只执行一次）
        if self.shuffle:
            video_indices = torch.randperm(
                self.num_videos, 
                generator=self.generator
            ).tolist()
        else:
            video_indices = list(range(self.num_videos))
        
        # 映射到实际的video_id
        video_ids = [self.valid_videos[i] for i in video_indices]
        
        # Step 2: 按videos_per_batch分组
        video_chunks = [
            video_ids[i : i + self.videos_per_batch]
            for i in range(0, len(video_ids), self.videos_per_batch)
        ]
        
        # Step 3: 丢弃不完整的batch（drop_last）
        video_chunks = [
            chunk for chunk in video_chunks 
            if len(chunk) == self.videos_per_batch
        ]
        
        # Step 4: 生成索引序列
        for chunk in video_chunks:  # 对每个batch
            for _ in range(self.repeat_count):  # 重复整个batch
                for video_id in chunk:  # batch内的每个视频
                    tasks_dict = self.video_task_map[video_id]
                    
                    # 按指定顺序输出每个任务
                    for task_name in self.task_names:
                        task_idx = tasks_dict[task_name]
                        
                        # 每个任务重复mini_repeat_count次
                        for _ in range(self.mini_repeat_count):
                            yield task_idx
    
    def __len__(self) -> int:
        """
        返回总样本数
        = (完整batch数) * (batch_size) * (mini_repeat_count) * (repeat_count)
        """
        num_complete_batches = self.num_videos // self.videos_per_batch
        return (
            num_complete_batches * 
            self.batch_size * 
            self.mini_repeat_count * 
            self.repeat_count
        )


class GRPOSequentialSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        # self.seed = seed
        # self.generator = torch.Generator()  # Create a local random generator
        # if seed is not None:
        #     self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        # print(index)
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[List[Union[PreTrainedModel, nn.Module]]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        patch_save_last_checkpoint()
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.args = args
        self.ref_adapter_name = getattr(args, 'ref_adapter_name', None)
        self.model_adapter_name = None
        self.is_multimodal = model.model_meta.is_multimodal

        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = identity_data_collator  # No data collation is needed in GRPO

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys() if not hasattr(model, 'get_base_model') else
            inspect.signature(model.get_base_model().forward).parameters.keys())

        self.vllm_client = kwargs.pop('vllm_client', None)
        self.chord_sft_dataset = kwargs.pop('chord_sft_dataset', None)
        reward_templates = kwargs.pop('reward_template', None)
        self._prepare_algorithm_params()
        super().__init__(model, ref_model, *_args, **kwargs)
        self._prepare_chord_dataset()
        self.prepare_rollout()
        self._prepare_rewards(reward_funcs, reward_model, reward_templates)

        if not self.reward_funcs and not self.use_gym_env:
            raise ValueError('You must specify reward_funcs or reward_model')

        if self.args.eval_strategy != 'no':
            total_eval_batch_size = self.args.per_device_eval_batch_size * \
                self.accelerator.num_processes // self.args.num_generations
            assert len(self.eval_dataset) >= total_eval_batch_size, (
                f'eval_dataset size {len(self.eval_dataset)} is smaller than '
                f'total_eval_batch_size {total_eval_batch_size}. '
                f'Please increase the size of eval_dataset or set a larger value for split_dataset_ratio.')

        self._prepare_liger_loss()
        self._prepare_metrics()

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if not self.args.use_vllm:
            from swift.llm import PtEngine
            infer_template = copy(self.template)
            infer_template.padding_free = False
            infer_template.sequence_parallel_size = 1
            self.engine = PtEngine.from_model_template(self.model, infer_template, max_batch_size=0)  # 0: no limit

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.args.dynamic_sample or self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()
        # flag indicating whether the evaluation has started
        self.eval_flag = False

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            self.args.gradient_accumulation_steps = self.args.gradient_accumulation_steps * sequence_parallel.world_size

        # for multi-turn server, maybe the num of rollout outputs is not equal to the num of rollout inputs
        self.dynamic_num_samples = False
        # Record the number of samples that need to be padded for even distribution across processes
        self.rollout_pad_count = 0

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # ======== MultiTask GRPO =========
        task_names = set()
        for item in self.train_dataset:
            if item.get("task", None) is not None:
                task_names.add(item["task"])
        self.task_names = list(task_names)
        self.use_paired = get_env_args('use_paired_sampler', bool, False)
        self.advantage_mode = get_env_args('advantage_mode', str, None)
        self.conditional_mode = get_env_args('conditional_mode', str, None)
        if self.use_paired:
            assert self.advantage_mode is not None, "Can not use paired_sampler without specifying advantage_mode"
            assert self.advantage_mode in ["independent", "grouped", "conditional"], f"Not supported advantage_mode: {self.advantage_mode}"
            if self.advantage_mode == "conditional":
                assert self.conditional_mode is not None, "Can not use advantage_mode without specifying conditional_mode"
            assert len(self.task_names) > 0, "Task names should not be empty when using paired_sampler"

        if self.conditional_mode == "default":
            self.task_coupling_fn = default_task_coupling
        elif self.conditional_mode == "gated":
            self.task_coupling_fn = gated_task_coupling
        elif self.conditional_mode == "hard_gated":
            self.task_coupling_fn = hard_gating_coupling
        elif self.conditional_mode == "hard_gated_bonus":
            self.task_coupling_fn = hard_gating_aigc_bonus_coupling
        elif self.conditional_mode == "bonus":
            self.task_coupling_fn = aigc_bonus_coupling

        self.use_sequential = get_env_args('use_sequential_sampler', bool, False)
            
            
    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            return RepeatSampler(
                data_source=train_dataset or self.train_dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation * sequence_parallel.world_size,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
        else:
            if self.use_paired:
                # 确保generation_batch_size可以被(2 * num_generations)整除: 因为每个视频有2个任务
                generation_batch_size = self.args.generation_batch_size
                num_generations = self.num_generations
                assert generation_batch_size % (2 * num_generations) == 0

                batch_size = generation_batch_size // num_generations
                return MultiTaskRepeatSampler(
                    data_source=train_dataset,
                    task_names=self.task_names,
                    mini_repeat_count=num_generations,
                    batch_size=batch_size,
                    repeat_count=self.num_iterations * self.args.steps_per_generation,
                    shuffle=self.shuffle_dataset,
                    seed=self.args.seed,
                )
            
            elif self.use_sequential:
                batch_size = self.args.generation_batch_size // self.num_generations
                # print("*"*500, "using GRPO sequential sampler")
                return GRPOSequentialSampler(
                    data_source=self.train_dataset,
                    mini_repeat_count=self.num_generations,
                    batch_size=batch_size,
                    repeat_count=self.num_iterations * self.args.steps_per_generation,
                    seed=self.args.seed,
                )
            else:
                # print(train_dataset)
                # print(train_dataset.column_names)
                # print(self.args.generation_batch_size, self.num_generations)
                return super()._get_train_sampler(train_dataset)

    @patch_profiling_decorator
    def _prepare_inputs(self, generation_batch: Dict[str, Union[torch.Tensor,
                                                                Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = 'train' if self.model.training else 'eval'
        if mode == 'train':
            num_rollout_samples = self.args.steps_per_generation * self.template.sequence_parallel_size
            generate_every = num_rollout_samples * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = generation_batch  # < this is the change
            inputs = self._buffered_inputs[self._step % num_rollout_samples]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @contextmanager
    def _template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        template.max_length = None
        try:
            yield
        finally:
            template.max_length = max_length

    def _generate_completions(self, inputs: DataType) -> DataType:
        # add prompt ids and system prompts
        ## TODO: if grpo_aug   use self_preprocess_inputs
        # print(inputs)
        inputs = self._preprocess_inputs(inputs)

        mode = 'train' if self.model.training else 'eval'
        if self.use_fast_infer:
            results = self._fast_infer(inputs)
        else:
            with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ), self.template.generate_context(), self.multi_turn_completion_length_context():
                results = self._infer_single_or_multi_turn(inputs, self.request_config)
                if mode == 'train':
                    # In training mode, ensure the model is returned to train() mode after inference
                    # This is necessary as pt engines set the model to eval mode during generation
                    self.model.train()
        return results

    @patch_profiling_decorator
    def _generate_and_score_completions(self, inputs: DataType) -> DataType:
        # resample for encoding failed data when set truncation_strategy 'delete'
        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs)

        inputs = self._generate_completions(inputs)
        # print("*"*80)
        # print(inputs[0].keys())
        # print(inputs[0]["messages"][-1]["content"])

        total_rewards_per_func = self._score_completions(inputs)
        mode = 'train' if self.model.training else 'eval'

        if self.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            inputs, total_rewards_per_func = self._dynamic_sampling(inputs, total_rewards_per_func)  # noqa

        batch_encoded_inputs = self._prepare_batch_inputs(inputs)

        total_advantages = self._compute_advantages(inputs, total_rewards_per_func, batch_encoded_inputs)

        local_advantages = get_even_process_data(self, total_advantages)
        # print(total_advantages.shape)   # [8] / [64]
        # print(len(local_advantages), len(inputs))   # 8 8 / 1 8
        # print(total_advantages, local_advantages)   # [8] [1] / [64] [8]
    
        assert len(local_advantages) == len(inputs)
        for i, advantage in enumerate(local_advantages):
            inputs[i]['advantages'] = advantage
        # log metrics in inputs
        self._logs['advantages'].extend(total_advantages.tolist())

        # Add advantages to each batch in batch_encoded_inputs
        gas_chunks = self.split_by_mini_batches(inputs)
        assert len(gas_chunks) == len(batch_encoded_inputs), \
            f'Mismatch: {len(gas_chunks)} chunks vs {len(batch_encoded_inputs)} batches'

        for batch, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
            if self.template.padding_free:
                lengths = batch_encoded['seq_lengths']
                advantages_stacked = torch.stack([data['advantages'] for data in batch])
                all_advantages = torch.repeat_interleave(advantages_stacked, lengths)
            else:
                all_advantages = torch.stack([data['advantages'] for data in batch])
            batch_encoded['advantages'] = all_advantages

        with patch_profiling_context(self, 'log_metrics'):
            # --- logs (prompts + completions) ---
            messages = [inp['messages'][:-1] for inp in inputs]
            completions = [deepcopy(inp['messages'][-1]['content']) for inp in inputs]
            for i, completion in enumerate(completions):
                if isinstance(completion, str):
                    continue
                if isinstance(completion, list):
                    token_ids = completion
                elif isinstance(completion, dict):
                    token_ids = completion['token_ids']
                completions[i] = self.processing_class.decode(token_ids)
            valid_messages = self._gather_and_flatten(messages, flatten_level=0)
            valid_completions = self._gather_and_flatten(completions, flatten_level=0)
            self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
            self._logs['completion'].extend(valid_completions)

            # Example: if you want to log extra data in the wandb / swanlab table,
            #          add them to metrics_to_gather
            # NOTE: every key you register must appear in ALL rollout outputs
            #       to avoid potential communication / synchronization issues
            metrics_for_logs_to_gather = {}

            if all('solution' in inp for inp in inputs):
                metrics_for_logs_to_gather['solution'] = [inp['solution'] for inp in inputs]

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                metrics_for_logs_to_gather['num_turns'] = [inp['rollout_infos']['num_turns'] for inp in inputs]

            if metrics_for_logs_to_gather:
                for key, value in metrics_for_logs_to_gather.items():
                    if key not in self._logs:
                        self._logs[key] = deque(maxlen=self.args.generation_batch_size)
                    self._logs[key].extend(self._gather_and_flatten(value, flatten_level=0))

        return batch_encoded_inputs

    @patch_profiling_decorator
    def _score_completions(self, inputs: DataType) -> torch.Tensor:
        """Score completions using all reward functions.

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with all reward values
        """
        device = self.accelerator.device
        # If using gym environment, extract rewards directly from inputs
        if self.use_gym_env:
            reward_from_gym = [inp['rollout_infos']['total_reward'] for inp in inputs]
            # For gym environment, there's only one total reward, so rewards_per_func is just local_rewards reshaped
            local_rewards_per_func = torch.tensor(
                reward_from_gym, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [num_examples, 1]
        else:
            # Compute rewards using reward functions
            local_rewards_per_func = self._compute_rewards_per_func(inputs)
            # print(self.accelerator.process_index, local_rewards_per_func, "@@@@@@@@@@@@")

        # gather rewards
        # print("&"*50, self.dynamic_num_samples)   # False
        if not self.dynamic_num_samples:
            total_rewards_per_func = gather(local_rewards_per_func)
            # print(self.accelerator.process_index, total_rewards_per_func, "#############")
        else:
            # gather_object to avoid shape mismatch
            local_rewards_list = [row.tolist() for row in local_rewards_per_func]
            total_rewards_per_func = gather_object(local_rewards_list)
            total_rewards_per_func = torch.tensor(
                total_rewards_per_func, dtype=torch.float32, device=self.accelerator.device)

        return total_rewards_per_func

    def _compute_rewards_per_func(self, inputs: DataType) -> torch.Tensor:
        """Compute rewards using all reward functions"""
        device = self.accelerator.device
        # print(self.accelerator.process_index, inputs, "==========")
        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in inputs]
        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with patch_profiling_context(self, reward_func_name):
                # reward model
                reward_kwargs = {'trainer_state': self.state}
                if self.enable_server_multi_turn:
                    trajectory_inputs = self._get_trajectory_inputs(inputs)
                    reward_kwargs.update({'trajectory_inputs': trajectory_inputs})
                if isinstance(reward_func, nn.Module):
                    output_reward_func = reward_model_plugin(inputs=inputs, **reward_kwargs)
                # reward function
                else:
                    # Repeat all input columns (but "messages" and "completion") to match the number of generations
                    reward_kwargs.update(RowPreprocessor.rows_to_batched(inputs))
                    output_reward_func = reward_func(completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs['completion'] = completions[nan_row_idx]
            logger.warning(f'All reward functions returned None for the following kwargs: {row_reward_kwargs}. '
                           'Please ensure that at least one reward function returns a valid reward.')

        return rewards_per_func

    def _compute_advantages(self, inputs: DataType, rewards_per_func: torch.Tensor,
                            batch_encoded_inputs: List[DataType]) -> torch.Tensor:
        """
        Compute advantages for RL training.

        Supports two modes:
        1. **Default grouped mode** (no prompt_ids / request_ids provided)
        - Assumes rewards are grouped by prompt and each prompt has exactly
            `self.num_generations` completions.
        - Computes advantages relative to group mean.
        2. **Request-aware mode** (multi-turn conversations, variable number of samples)
        - Groups rewards by unique `request_id` and computes statistics per prompt.
        - Handles dynamic sample sizes where multiple request_ids may share the same prompt.

        Args:
            inputs (DataType):
                Input data samples.
            rewards_per_func (torch.Tensor):
                Reward values for each reward function, shape `(N, num_reward_funcs)`.

        Returns:
            **advantages** (torch.Tensor):
                Computed advantages, shape `(N,)`.
        """
        ## ============= Coupled Advantage Mode =============
        if self.use_paired:
            return self._compute_coupled_advantages(inputs, rewards_per_func, batch_encoded_inputs)

        def normalize_advantages(advantages: torch.Tensor, rewards_std: torch.Tensor) -> torch.Tensor:
            """Normalize advantages if configured; otherwise, return as-is."""
            if self.scale_rewards != 'none':
                return advantages / (rewards_std + 1e-4)
            return advantages

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, self.num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*self.num_generations, self.num_reward_funcs]
            mode = 'train' if self.model.training else 'eval'
            group_rewards = rewards.view(-1, self.num_generations)
            rewards_mean = group_rewards.mean(-1).mean().item()
            if self.scale_rewards in ['group', 'none']:
                rewards_std = group_rewards.std(-1).mean().item()
            elif self.scale_rewards == 'batch':
                rewards_std = rewards.std().item()
            is_std_zero = torch.isclose(group_rewards.std(dim=1), torch.zeros_like(group_rewards.std(dim=1)))

            self._metrics[mode]['reward'].append(rewards_mean)
            self._metrics[mode]['reward_std'].append(rewards_std)
            self._metrics[mode]['frac_reward_zero_std'].append(is_std_zero.float().mean().item())

            # Log per-reward-function statistics using deduplicated rewards_per_func
            for i, name in enumerate(self.reward_func_names):
                col = rewards_per_func_for_metrics[:, i]
                self._metrics[mode][f'rewards/{name}/mean'].append(torch.nanmean(col).item())
                self._metrics[mode][f'rewards/{name}/std'].append(nanstd(col).item())

        def log_rewards_all(rewards_per_func: torch.Tensor):
            """Log all rewards for debugging."""
            for i, name in enumerate(self.reward_func_names):
                self._logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

        # Step 0. Aggregate final reward using reward weights
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        if self.kl_in_reward and self.beta != 0.0:
            kl_list = []
            for batch_encoded in batch_encoded_inputs:
                old_per_token_logps = batch_encoded['old_per_token_logps']
                ref_per_token_logps = batch_encoded['ref_per_token_logps']
                completion_mask = batch_encoded['completion_mask']
                if self.template.padding_free:
                    lengths = batch_encoded['seq_lengths']
                    per_token_kl = torch.split(old_per_token_logps - ref_per_token_logps, lengths.tolist(), dim=1)
                    completion_masks = torch.split(completion_mask, lengths.tolist(), dim=1)
                    kl = torch.cat([(kl * mask).sum(-1) for kl, mask in zip(per_token_kl, completion_masks)])
                else:
                    per_token_kl = old_per_token_logps - ref_per_token_logps
                    kl = (per_token_kl * completion_mask).sum(-1)
                kl_list.append(kl)

            kl = torch.cat(kl_list, dim=0)
            kl = gather(kl)
            mode = 'train' if self.model.training else 'eval'
            self._metrics[mode]['kl'].append(kl.nanmean().item())
            rewards = rewards - self.beta * kl

        # --------------------------------------------------
        # Case 1: Default grouped mode
        # --------------------------------------------------
        if not self.dynamic_num_samples:
            grouped_rewards = rewards.view(-1, self.num_generations)
            K = self.num_generations

            # Compute group statistics
            group_rewards_mean = grouped_rewards.mean(dim=1)

            # Broadcast stats back to the original shape
            group_rewards_mean = group_rewards_mean.repeat_interleave(K)

            # Compute advantages based on estimation type
            if self.advantage_estimator == 'rloo':
                # RLOO: Leave-One-Out baseline
                # A_i = r_i - mean(r_j for j != i)
                # = r_i * K/(K-1) - mean_all * K/(K-1)
                advantages = rewards * K / (K - 1) - group_rewards_mean * K / (K - 1)
            else:  # 'grpo' or 'reinforce_plus_plus'
                # Both use group mean as baseline
                advantages = rewards - group_rewards_mean

            # Normalize advantages based on estimator and scale_rewards
            if self.advantage_estimator == 'reinforce_plus_plus':
                # REINFORCE++: Use std of advantages (not rewards)
                if self.scale_rewards == 'batch':
                    # Global whitening: std computed on advantages
                    # Note: advantages.mean() is mathematically 0, no need to subtract
                    advantages_std = advantages.std().expand_as(advantages)
                elif self.scale_rewards == 'group':
                    # Group-level whitening on advantages
                    advantages_grouped = advantages.view(-1, K)
                    advantages_std = advantages_grouped.std(dim=1).repeat_interleave(K)
                else:  # 'none'
                    advantages_std = None
                if advantages_std is not None:
                    advantages = normalize_advantages(advantages, advantages_std)
            else:  # 'grpo' or 'rloo'
                # GRPO/RLOO: Use std of original rewards
                if self.scale_rewards == 'batch':
                    rewards_std = rewards.std().expand_as(rewards)
                elif self.scale_rewards == 'group':
                    rewards_std = grouped_rewards.std(dim=1).repeat_interleave(K)
                else:  # 'none'
                    rewards_std = None
                if rewards_std is not None:
                    advantages = normalize_advantages(advantages, rewards_std)

            # Log metrics once per group
            log_rewards_metrics(rewards=grouped_rewards, rewards_per_func_for_metrics=rewards_per_func)

            # Log all rewards
            log_rewards_all(rewards_per_func)

            return advantages

        # --------------------------------------------------
        # Case 2: Request-aware mode
        # --------------------------------------------------
        else:
            prompt_ids = gather_object([inp['prompt_id'] for inp in inputs])
            request_ids = gather_object([inp['request_id'] for inp in inputs])
            assert rewards.shape[0] == len(prompt_ids) == len(request_ids)
            device = self.accelerator.device

            # Step 1. Deduplicate request_ids
            unique_indices = self._get_last_indices(request_ids)
            unique_request_ids = [request_ids[i] for i in unique_indices.cpu()]
            unique_prompt_ids = [prompt_ids[i] for i in unique_indices.cpu()]

            # Step 2. Validate rewards consistency within the same request_id
            for rid in set(request_ids):
                idxs = [i for i, r in enumerate(request_ids) if r == rid]
                if not torch.allclose(rewards[idxs], rewards[idxs[0]].expand(len(idxs)), atol=1e-6):
                    raise ValueError(f'Inconsistent rewards detected for request_id={rid}.')

            # Step 3. Group rewards by prompt_id and compute prompt-level mean/std
            unique_rewards = rewards[unique_indices]
            prompt_to_indices = {}
            for idx, pid in enumerate(unique_prompt_ids):
                prompt_to_indices.setdefault(pid, []).append(idx)

            prompt_means = torch.zeros(len(unique_rewards), device=device)
            for pid, idxs in prompt_to_indices.items():
                idx_tensor = torch.tensor(idxs, device=device)
                r_group = unique_rewards[idx_tensor]
                prompt_means[idx_tensor] = r_group.mean()

            # Step 4. Compute advantages
            if self.advantage_estimator == 'rloo':
                # RLOO: Leave-One-Out baseline for dynamic mode
                request_advantages = torch.zeros_like(unique_rewards)
                for pid, idxs in prompt_to_indices.items():
                    K = len(idxs)
                    idx_tensor = torch.tensor(idxs, device=device)
                    r_group = unique_rewards[idx_tensor]
                    # A_i = r_i * K/(K-1) - mean * K/(K-1)
                    request_advantages[idx_tensor] = (r_group * K / (K - 1) - r_group.mean() * K / (K - 1))
            else:  # 'grpo' or 'reinforce_plus_plus'
                # Both use group mean as baseline
                request_advantages = unique_rewards - prompt_means

            # Step 5. Normalize advantages
            if self.advantage_estimator == 'reinforce_plus_plus':
                # REINFORCE++: Use std of advantages (not rewards)
                if self.scale_rewards == 'batch':
                    # Global whitening: std computed on advantages
                    # Note: advantages.mean() is mathematically 0, no need to subtract
                    advantages_std = request_advantages.std()
                    prompt_stds = torch.full_like(request_advantages, advantages_std)
                elif self.scale_rewards == 'group':
                    # Group-level whitening on advantages
                    prompt_stds = torch.zeros(len(unique_rewards), device=device)
                    for pid, idxs in prompt_to_indices.items():
                        idx_tensor = torch.tensor(idxs, device=device)
                        adv_group = request_advantages[idx_tensor]
                        prompt_stds[idx_tensor] = adv_group.std()
                else:  # 'none'
                    prompt_stds = None
                if prompt_stds is not None:
                    request_advantages = normalize_advantages(request_advantages, prompt_stds)
            else:  # 'grpo' or 'rloo'
                # GRPO/RLOO: Use std of original rewards
                if self.scale_rewards == 'batch':
                    rewards_std = unique_rewards.std()
                    prompt_stds = torch.full_like(unique_rewards, rewards_std)
                elif self.scale_rewards == 'group':
                    prompt_stds = torch.zeros(len(unique_rewards), device=device)
                    for pid, idxs in prompt_to_indices.items():
                        idx_tensor = torch.tensor(idxs, device=device)
                        r_group = unique_rewards[idx_tensor]
                        prompt_stds[idx_tensor] = r_group.std()
                else:  # 'none'
                    prompt_stds = None
                if prompt_stds is not None:
                    request_advantages = normalize_advantages(request_advantages, prompt_stds)

            # Map advantages back to original order
            rid_to_idx = {rid: idx for idx, rid in enumerate(unique_request_ids)}
            indices_in_unique = torch.tensor([rid_to_idx[r] for r in request_ids], device=device)
            advantages = request_advantages[indices_in_unique]

            # Step 5. Log metrics for unique request_ids
            log_rewards_metrics(rewards=unique_rewards, rewards_per_func_for_metrics=rewards_per_func[unique_indices])

            # Step 6. Log all rewards
            log_rewards_all(rewards_per_func)

            return advantages

    # ===========-----------------------------=============== Custom Start ============--------------------------------------================
    def _compute_coupled_advantages(
        self, 
        inputs: List[Dict], 
        rewards_per_func: torch.Tensor,
        batch_encoded_inputs: List[Dict]
    ) -> torch.Tensor:
        """
        多任务版本的优势计算
        
        Args:
            inputs: 输入样本列表，每个包含 video_id, task, request_id 等
            rewards_per_func: shape [N, num_reward_funcs]
                堆叠形式: [[样本1_任务1_r1, ...], [样本1_任务2_r1, ...], ...]
            batch_encoded_inputs: 编码后的输入（用于KL散度）
        
        Returns:
            advantages: shape [N]
        """
        all_inputs = self._gather_and_flatten(inputs, flatten_level=0)
        
        # Step 0: 聚合最终奖励（使用reward weights）
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        
        # Step 0.1: 添加KL惩罚（如果需要）
        if self.kl_in_reward and self.beta != 0.0:
            kl = self._compute_kl_penalty(batch_encoded_inputs)
            rewards = rewards - self.beta * kl
        
        # Step 1: 解析批次结构，构建视频-任务映射
        # video_task_structure = self._build_video_task_structure(inputs, rewards)
        video_task_structure = self._build_video_task_structure(all_inputs, rewards)
        
        # Step 2: 根据模式选择优势计算方法
        if self.advantage_mode == 'independent':
            # 标准GRPO：每个任务独立计算
            advantages = self._compute_independent_advantages(
                rewards, video_task_structure
            )
        elif self.advantage_mode == 'grouped':
            # 耦合模式：同一视频的任务一起考虑
            advantages = self._compute_grouped_advantages(
                rewards, rewards_per_func, video_task_structure
            )
        elif self.advantage_mode == 'conditional':
            # 条件模式：一个任务影响另一个
            advantages = self._compute_conditional_advantages(
                rewards, rewards_per_func, video_task_structure
            )
        else:
            raise ValueError(f"Unknown task_advantage_mode: {self.advantage_mode}")
        
        # Step 3: 日志记录
        self._log_multitask_rewards_metrics(
            rewards, rewards_per_func, video_task_structure
        )
        
        return advantages
    
    def _build_video_task_structure(
        self, 
        inputs: List[Dict], 
        rewards: torch.Tensor
    ) -> Dict[str, Any]:
        """
        构建视频-任务的层级结构（通用版本）
        
        自动适应任意数量和名称的任务
        
        Returns:
            structure: {
                'video_to_samples': {
                    'video_0': {
                        'tasks': {
                            'task_name_1': [idx0, idx1, ...],
                            'task_name_2': [idx2, idx3, ...],
                            ...
                        },
                        'all_indices': [idx0, idx1, idx2, ...],
                        'num_generations_per_task': {'task1': 2, 'task2': 2},
                        'total_samples': int,
                    },
                    ...
                },
                'sample_to_video': [video_id for each sample],
                'sample_to_task': [task for each sample],
                'task_names': ['task1', 'task2', ...],
                'num_videos': int,
                'num_samples': int,
                'num_tasks': int,
            }
        """
        num_samples = len(inputs)
        device = rewards.device
        
        # 提取关键信息
        sample_to_video = []
        sample_to_task = []
        
        # 动态发现所有任务名
        discovered_tasks = set()
        
        # 初始化视频字典（使用通用结构）
        video_to_samples = defaultdict(lambda: {
            'tasks': defaultdict(list),  # 动态任务字典
            'all_indices': [],
        })
        
        # 遍历所有样本，构建映射
        for idx, inp in enumerate(inputs):
            video_id = inp['video_id']
            task = inp['task']
            
            sample_to_video.append(video_id)
            sample_to_task.append(task)
            
            # 动态添加任务
            video_to_samples[video_id]['tasks'][task].append(idx)
            video_to_samples[video_id]['all_indices'].append(idx)
            discovered_tasks.add(task)
        
        # 确定最终的任务列表
        if self.task_names is None:
            # 从数据推断
            final_task_names = sorted(list(discovered_tasks))
            if self._inferred_task_names is None:
                self._inferred_task_names = final_task_names
                print(f"📋 自动检测到的任务: {final_task_names}")
            elif set(final_task_names) != set(self._inferred_task_names):
                print(f"⚠️  警告: 任务列表变化: {self._inferred_task_names} -> {final_task_names}")
        else:
            # 使用指定的任务列表
            final_task_names = self.task_names
            missing_tasks = discovered_tasks - set(final_task_names)
            if missing_tasks:
                print(f"⚠️  警告: 发现未指定的任务: {missing_tasks}")
        
        # 计算每个视频每个任务的生成次数
        for video_id, video_info in video_to_samples.items():
            num_gens_per_task = {}
            for task_name in final_task_names:
                num_gens = len(video_info['tasks'].get(task_name, []))
                num_gens_per_task[task_name] = num_gens
            
            video_info['num_generations_per_task'] = num_gens_per_task
            video_info['total_samples'] = len(video_info['all_indices'])

        # 构建最终结构
        structure = {
            'video_to_samples': dict(video_to_samples),
            'sample_to_video': sample_to_video,
            'sample_to_task': sample_to_task,
            'task_names': final_task_names,
            'num_videos': len(video_to_samples),
            'num_samples': num_samples,
            'num_tasks': len(final_task_names),
        }
        
        return structure
    
    def _compute_independent_advantages(
        self,
        rewards: torch.Tensor,
        structure: Dict[str, Any]
    ) -> torch.Tensor:
        """
        独立模式：每个任务独立计算优势（通用版本）
        """
        device = rewards.device
        num_samples = structure['num_samples']
        task_names = structure['task_names']
        advantages = torch.zeros(num_samples, device=device)
        
        # 遍历所有视频
        for video_id, video_info in structure['video_to_samples'].items():
            
            # 遍历所有任务类型
            for task_name in task_names:
                task_indices = video_info['tasks'].get(task_name, [])
                
                if len(task_indices) == 0:
                    continue  # 该视频没有这个任务
                
                K = len(task_indices)
                task_rewards = rewards[task_indices]
                
                # 计算组均值
                task_mean = task_rewards.mean()
                
                # 计算优势
                if self.advantage_estimator == 'rloo':
                    task_advantages = (
                        task_rewards * K / (K - 1) - 
                        task_mean * K / (K - 1)
                    )
                else:
                    task_advantages = task_rewards - task_mean
                
                # 归一化
                if self.scale_rewards == 'group':
                    task_std = task_rewards.std() + 1e-4
                    task_advantages = task_advantages / task_std
                
                # 写回
                for i, idx in enumerate(task_indices):
                    advantages[idx] = task_advantages[i]
        
        # 全局归一化
        if self.scale_rewards == 'batch':
            advantages_std = advantages.std() + 1e-4
            advantages = advantages / advantages_std
        
        return advantages
    
    def _compute_grouped_advantages(
        self,
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
        structure: Dict[str, Any]
    ) -> torch.Tensor:
        """
        耦合模式：同一视频的所有任务一起计算（通用版本）
        """
        device = rewards.device
        num_samples = structure['num_samples']
        advantages = torch.zeros(num_samples, device=device)
        
        for video_id, video_info in structure['video_to_samples'].items():
            all_indices = video_info['all_indices']
            K: int = len(all_indices)
            
            # 提取该视频的所有奖励
            video_rewards = rewards[all_indices]
            
            # 计算视频级别的统计量
            video_mean = video_rewards.mean()
            video_std = video_rewards.std() + 1e-4
            
            # 计算优势
            if self.advantage_estimator == 'rloo':
                video_advantages = (
                    video_rewards * K / (K - 1) - 
                    video_mean * K / (K - 1)
                )
            else:
                video_advantages = video_rewards - video_mean
            
            # 归一化
            if self.scale_rewards == 'group':
                video_advantages = video_advantages / video_std
            
            # 写回
            for i, idx in enumerate(all_indices):
                advantages[idx] = video_advantages[i]
        
        # 全局归一化
        if self.scale_rewards == 'batch':
            advantages_std = advantages.std() + 1e-4
            advantages = advantages / advantages_std
        
        return advantages

    def _compute_conditional_advantages(
        self,
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
        structure: Dict[str, Any]
    ) -> torch.Tensor:
        """
        条件模式：支持任意任务间的相互影响（通用版本）
        """
        device = rewards.device
        num_samples = structure['num_samples']
        task_names = structure['task_names']
        
        # Step 1: 计算基础优势
        base_advantages = self._compute_independent_advantages(rewards, structure)
        
        # Step 2: 应用通用耦合函数
        adjusted_advantages = torch.zeros_like(base_advantages)
        
        
        for video_id, video_info in structure['video_to_samples'].items():
            # 收集该视频所有任务的信息
            task_data = {}
            for task_name in task_names:
                task_indices = video_info['tasks'].get(task_name, [])
                if len(task_indices) > 0:
                    task_data[task_name] = {
                        'indices': task_indices,
                        'rewards': rewards[task_indices],
                        'rewards_per_func': rewards_per_func[task_indices],
                        'base_advantages': base_advantages[task_indices],
                    }
            
            # 应用通用耦合函数
            adjusted_task_data = self.task_coupling_fn(task_data)
            
            # 写回调整后的优势
            for task_name, data in adjusted_task_data.items():
                for i, idx in enumerate(data['indices']):
                    adjusted_advantages[idx] = data['adjusted_advantages'][i]
        
        return adjusted_advantages
    
    def _compute_kl_penalty(self, batch_encoded_inputs: List[Dict]) -> torch.Tensor:
        """计算KL散度惩罚（复用原有逻辑）"""
        kl_list = []
        for batch_encoded in batch_encoded_inputs:
            old_per_token_logps = batch_encoded['old_per_token_logps']
            ref_per_token_logps = batch_encoded['ref_per_token_logps']
            completion_mask = batch_encoded['completion_mask']
            
            if self.template.padding_free:
                lengths = batch_encoded['seq_lengths']
                per_token_kl = torch.split(
                    old_per_token_logps - ref_per_token_logps, 
                    lengths.tolist(), 
                    dim=1
                )
                completion_masks = torch.split(completion_mask, lengths.tolist(), dim=1)
                kl = torch.cat([
                    (kl * mask).sum(-1) 
                    for kl, mask in zip(per_token_kl, completion_masks)
                ])
            else:
                per_token_kl = old_per_token_logps - ref_per_token_logps
                kl = (per_token_kl * completion_mask).sum(-1)
            
            kl_list.append(kl)
        
        kl = torch.cat(kl_list, dim=0)
        kl = gather(kl)
        
        # 记录指标
        mode = 'train' if self.model.training else 'eval'
        self._metrics[mode]['kl'].append(kl.nanmean().item())
        
        return kl
    
    def _log_multitask_rewards_metrics(
        self,
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
        structure: Dict[str, Any]
    ):
        """
        记录多任务的奖励指标（通用版本，保持原始记录逻辑）
        
        Args:
            rewards: shape [num_samples] 聚合后的奖励
            rewards_per_func: shape [num_samples, num_reward_funcs] 各reward function的奖励
            structure: _build_video_task_structure返回的结构
        
        记录内容：
        1. 整体奖励统计（保持原始格式）
        2. 分任务的奖励统计
        3. 视频级别的奖励分布
        4. 按reward function统计（整体+分任务）
        5. 所有原始奖励（用于调试）
        """
        mode = 'train' if self.model.training else 'eval'
        device = rewards.device
        
        task_names = structure['task_names']
        num_videos = structure['num_videos']
        num_tasks = structure['num_tasks']
        
        # ========== 1. 整体统计（保持原始格式） ==========
        
        # 重构rewards为分组形式 [num_videos, num_tasks * num_generations]
        # 这里假设每个视频每个任务的生成次数相同
        video_rewards_list = []
        for video_id, video_info in structure['video_to_samples'].items():
            all_indices = video_info['all_indices']
            video_rewards_list.append(rewards[all_indices])
        
        if len(video_rewards_list) > 0:
            # 尝试堆叠（假设每个视频的样本数相同）
            try:
                grouped_rewards = torch.stack(video_rewards_list, dim=0)  # [num_videos, samples_per_video]
                
                # 计算每组的均值和标准差
                rewards_mean = grouped_rewards.mean(-1).mean().item()
                
                if self.scale_rewards in ['group', 'none']:
                    rewards_std = grouped_rewards.std(-1).mean().item()
                elif self.scale_rewards == 'batch':
                    rewards_std = rewards.std().item()
                else:
                    rewards_std = rewards.std().item()
                
                # 检查方差为零的比例（表示该组内所有生成的奖励相同）
                is_std_zero = torch.isclose(
                    grouped_rewards.std(dim=1),
                    torch.zeros_like(grouped_rewards.std(dim=1))
                )
                frac_zero_std = is_std_zero.float().mean().item()
                
            except RuntimeError:
                # 如果样本数不一致，退化为简单统计
                rewards_mean = rewards.mean().item()
                rewards_std = rewards.std().item()
                frac_zero_std = 0.0
        else:
            rewards_mean = 0.0
            rewards_std = 0.0
            frac_zero_std = 0.0
        
        # 记录整体指标（与原始格式一致）
        self._metrics[mode]['reward'].append(rewards_mean)
        self._metrics[mode]['reward_std'].append(rewards_std)
        self._metrics[mode]['frac_reward_zero_std'].append(frac_zero_std)
        
        # ========== 2. 按任务统计 ==========
        task_indices_map = {}  # {task_name: [indices]}
        
        for task_name in task_names:
            # 收集该任务的所有样本索引
            task_indices = []
            for video_id, video_info in structure['video_to_samples'].items():
                task_samples = video_info['tasks'].get(task_name, [])
                task_indices.extend(task_samples)
            
            task_indices_map[task_name] = task_indices
            
            if len(task_indices) > 0:
                task_rewards = rewards[task_indices]
                
                # 记录该任务的统计量（命名格式：reward/{task_name}_mean）
                self._metrics[mode][f'reward/{task_name}_mean'].append(
                    task_rewards.mean().item()
                )
                self._metrics[mode][f'reward/{task_name}_std'].append(
                    task_rewards.std().item()
                )
        
        # ========== 3. 视频级别统计 ==========
        # video_level_rewards = []
        # for video_id, video_info in structure['video_to_samples'].items():
        #     all_indices = video_info['all_indices']
        #     if len(all_indices) > 0:
        #         video_reward_mean = rewards[all_indices].mean().item()
        #         video_level_rewards.append(video_reward_mean)
        
        # if len(video_level_rewards) > 0:
        #     video_rewards_tensor = torch.tensor(
        #         video_level_rewards,
        #         dtype=torch.float32,
        #         device=device
        #     )
        #     self._metrics[mode]['reward/video_mean'].append(
        #         video_rewards_tensor.mean().item()
        #     )
        #     self._metrics[mode]['reward/video_std'].append(
        #         video_rewards_tensor.std().item()
        #     )
        
        # ========== 4. 按reward function统计 ==========
        for func_idx, func_name in enumerate(self.reward_func_names):
            func_rewards = rewards_per_func[:, func_idx]
            
            # 4.1 整体统计（所有任务）
            # self._metrics[mode][f'rewards/{func_name}/mean'].append(
            #     torch.nanmean(func_rewards).item()
            # )
            # self._metrics[mode][f'rewards/{func_name}/std'].append(
            #     nanstd(func_rewards).item()
            # )
            
            # 4.2 分任务统计
            for task_name in task_names:
                task_indices = task_indices_map.get(task_name, [])
                
                if len(task_indices) > 0:
                    task_func_rewards = func_rewards[task_indices]
                    
                    # 命名格式：rewards/{func_name}/{task_name}_mean
                    self._metrics[mode][f'rewards/{func_name}/{task_name}_mean'].append(
                        torch.nanmean(task_func_rewards).item()
                    )
        
        # ========== 5. 记录所有原始奖励（用于调试） ==========
        for func_idx, func_name in enumerate(self.reward_func_names):
            if func_name not in self._logs['rewards']:
                self._logs['rewards'][func_name] = []
            
            self._logs['rewards'][func_name].extend(
                rewards_per_func[:, func_idx].tolist()
            )
    # ===========-----------------------------=============== Custom End ============--------------------------------------================

    @patch_profiling_decorator
    def _dynamic_sampling(self, inputs, rewards_per_func):
        """
        Perform dynamic sampling to replace samples with zero-reward-variance groups.

        This method implements DAPO (https://arxiv.org/abs/2503.14476) by replacing
        samples from groups with zero reward variance (std=0) through resampling.

        Args:
            inputs: local input data samples
            rewards_per_func: reward per function for global data samples

        Returns:
            tuple: (inputs, rewards_per_func) with zero-variance groups replaced by resampled data
        """
        # DAPO https://arxiv.org/abs/2503.14476
        # Replaces samples with zero-reward-variance groups (std=0)
        resample_count = 0
        valid_samples = []
        valid_rewards_per_func = []
        origin_data = (inputs, rewards_per_func)

        while resample_count < self.max_resample_times:
            rewards_std = self.compute_std(inputs, rewards_per_func)
            valid_mask = (rewards_std > 0)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            if len(valid_samples) >= self.args.generation_batch_size:
                break

            inputs = next(self.dynamic_resample_iterator)
            if self.template.truncation_strategy == 'raise':
                inputs = self.resample_encode_failed_inputs(inputs)
            inputs = Trainer._prepare_inputs(self, inputs)
            inputs = self._generate_completions(inputs)
            rewards_per_func = self._score_completions(inputs)
            resample_count += 1

        if len(valid_samples) >= self.args.generation_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = valid_samples[:self.args.generation_batch_size][process_slice]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.args.generation_batch_size]
        else:
            logger.warning(f'There are still std=0 groups present after {self.max_resample_times} retries.')
            inputs, rewards_per_func = origin_data

        return inputs, rewards_per_func

    def compute_std(self, inputs: DataType, rewards_per_func: torch.Tensor) -> torch.Tensor:
        """Compute the standard deviation of the rewards per function."""
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        if not self.dynamic_num_samples:
            grouped_rewards = rewards.view(-1, self.num_generations)
            group_rewards_std = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations)
            return group_rewards_std
        else:
            prompt_ids = gather_object([inp['prompt_id'] for inp in inputs])
            request_ids = gather_object([inp['request_id'] for inp in inputs])
            device = self.accelerator.device
            unique_indices = self._get_last_indices(request_ids)
            unique_request_ids = [request_ids[i] for i in unique_indices.cpu()]
            unique_prompt_ids = [prompt_ids[i] for i in unique_indices.cpu()]

            unique_rewards = rewards[unique_indices]
            prompt_to_indices = {}
            for idx, pid in enumerate(unique_prompt_ids):
                prompt_to_indices.setdefault(pid, []).append(idx)

            prompt_stds = torch.zeros(len(unique_rewards), device=device)
            for pid, idxs in prompt_to_indices.items():
                idx_tensor = torch.tensor(idxs, device=device)
                r_group = unique_rewards[idx_tensor]
                prompt_stds[idx_tensor] = r_group.std()
            rid_to_idx = {rid: idx for idx, rid in enumerate(unique_request_ids)}
            indices_in_unique = torch.tensor([rid_to_idx[r] for r in request_ids], device=device)
            rewards_std = prompt_stds[indices_in_unique]

            return rewards_std

    def split_by_mini_batches(self, inputs: DataType) -> List[DataType]:
        """
        Split inputs into mini-batches, handling variable generation counts.

        When rollout count differs from expected (bs * spg * num_generations),
        we need to adjust the splitting logic to maintain proper batch sizes.

        This method divides the input data into chunks based on the steps per generation (spg).
        If the total number of inputs is not evenly divisible by spg, the remainder is
        distributed across the first few chunks to ensure all data is included.

        Args:
            inputs (DataType): List of input data samples to be split into mini-batches.

        Returns:
            List[DataType]: A list of data chunks, where each chunk represents one step
                           in the generation process. The number of chunks equals spg.
        """
        # Slice to keep only the local part of the data
        if self.template.sequence_parallel_size == 1:
            mode: str = 'train' if self.model.training else 'eval'
            spg: int = self.args.steps_per_generation if mode == 'train' else 1

            chunk_size: int = len(inputs) // spg
            remainder: int = len(inputs) % spg
            spg_chunks: List[DataType] = []

            start_idx: int = 0
            for i in range(spg):
                current_chunk_size: int = chunk_size + (1 if i < remainder else 0)
                end_idx: int = start_idx + current_chunk_size
                spg_chunks.append(inputs[start_idx:end_idx])
                start_idx = end_idx

            return spg_chunks
        else:
            from swift.trainers.sequence_parallel import sequence_parallel
            """Split by mini batches for GRPO sequence parallel training"""
            output = [None] * sequence_parallel.sp_world_size
            # gather inputs within a sp group
            dist.all_gather_object(output, inputs, group=sequence_parallel.sp_group)
            if sequence_parallel.rp_world_size > 1:
                output_rp = [None] * sequence_parallel.rp_world_size
                output = [p for sublist in output for p in sublist]
                dist.all_gather_object(output_rp, output, group=sequence_parallel.rp_group)
                output = output_rp
            output = [p for sublist in output for p in sublist]
            inputs = output

            mode = 'train' if self.model.training else 'eval'
            spg = self.args.steps_per_generation * sequence_parallel.world_size if mode == 'train' else 1

            if mode == 'eval':
                # TODO only take the first bs rows, because eval does not support loop
                bs = self.args.per_device_eval_batch_size
                inputs = inputs[:bs]
                spg = 1

            # Use the new dynamic splitting logic
            chunk_size: int = len(inputs) // spg
            remainder: int = len(inputs) % spg
            spg_chunks: List[DataType] = []

            start_idx: int = 0
            for i in range(spg):
                current_chunk_size: int = chunk_size + (1 if i < remainder else 0)
                end_idx: int = start_idx + current_chunk_size
                spg_chunks.append(inputs[start_idx:end_idx])
                start_idx = end_idx

            spg_chunks = to_device(spg_chunks, device=self.accelerator.device)
            return spg_chunks

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if is_peft_model(
                self.model) and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or 'default')

    @patch_profiling_decorator
    def _prepare_batch_inputs(self, inputs: DataType) -> List[DataType]:
        """
        Prepare the final batch inputs with ref/old_policy logps and other fields for RL training.

        Args:
            inputs (DataType): List of local input samples.

        Returns:
            List[DataType]: A list of prepared batch inputs, organized as [spg][bs]
        """
        template = self.template
        gas_chunks = self.split_by_mini_batches(inputs)
        ga_batch_encoded_inputs = []
        for batch in gas_chunks:
            # Encode and process each batch (size=bs)
            with self._template_context(template):
                for data in batch:
                    if 'response_token_ids' in data and data['response_token_ids']:
                        loss_mask = None
                        if 'response_loss_mask' in data and data['response_loss_mask']:
                            loss_mask = data['response_loss_mask']
                        data['messages'] = replace_assistant_response_with_ids(data['messages'],
                                                                               data['response_token_ids'], loss_mask)
                batch_encoded_inputs = [template.encode(data, return_length=True) for data in batch]
                batch_encoded_inputs = to_device(template.data_collator(batch_encoded_inputs), self.model.device)
                if self.dynamic_num_samples and self.is_multimodal:
                    batch_encoded_inputs['_origin_data'] = batch

            # Process labels and masks
            labels = batch_encoded_inputs.pop('labels')
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            extra_kwargs = {
                'completion_mask':
                labels[:, -logits_to_keep:] != -100,
                'truncated_mask':
                torch.tensor([b['is_truncated'] for b in batch], dtype=torch.bool, device=self.accelerator.device),
                'logits_to_keep':
                logits_to_keep,
            }
            if self.template.padding_free:
                position_ids = batch_encoded_inputs.get('text_position_ids')
                if position_ids is None:
                    position_ids = batch_encoded_inputs.get('position_ids')
                position_ids = position_ids.squeeze()
                assert position_ids is not None
                lengths = torch.diff(
                    torch.cat([(position_ids == 0).nonzero(as_tuple=True)[0],
                               torch.tensor([len(position_ids)]).to(position_ids.device)]))
                total_lengths = lengths.sum()
                # The first sentence has its prompt portion removed due to logits_to_keep
                lengths[0] = lengths[0] - (total_lengths - logits_to_keep)
                extra_kwargs.update({'seq_lengths': lengths})
            batch_encoded_inputs.update(extra_kwargs)

            with torch.no_grad():
                batch_encoded_inputs['old_per_token_logps'] = (
                    self._get_per_token_logps_and_entropies(self.model, batch_encoded_inputs)[0]
                    if self.old_policy() or self.kl_in_reward else None)
                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = \
                        self._get_per_token_logps_and_entropies(self.ref_model, batch_encoded_inputs)[0]
                else:
                    with self.null_ref_context():
                        ref_per_token_logps = \
                            self._get_per_token_logps_and_entropies(self.model, batch_encoded_inputs)[0]
                batch_encoded_inputs['ref_per_token_logps'] = ref_per_token_logps

            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        # --- log completion lengths ---
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device
        if self.template.padding_free:
            local_lengths = [inp['seq_lengths'].tolist() for inp in ga_batch_encoded_inputs]
        else:
            local_lengths = [inp['completion_mask'].sum(1).tolist() for inp in ga_batch_encoded_inputs]
        total_lengths = self._gather_and_flatten(local_lengths, dtype=torch.float32, device=device, flatten_level=1)

        # Store num_items_in_batch for DAPO loss (total completion tokens across all processes)
        num_items_in_batch = total_lengths.sum()
        for batch_encoded in ga_batch_encoded_inputs:
            batch_encoded['num_items_in_batch'] = num_items_in_batch

        self._metrics[mode]['completions/mean_length'].append(total_lengths.mean().item())
        self._metrics[mode]['completions/min_length'].append(total_lengths.min().item())
        self._metrics[mode]['completions/max_length'].append(total_lengths.max().item())

        # --- log completion clipped ratio ---
        local_trunc_masks = [inp['truncated_mask'].tolist() for inp in ga_batch_encoded_inputs]
        total_trunc_masks = self._gather_and_flatten(
            local_trunc_masks, dtype=torch.bool, device=device, flatten_level=1)

        if not self.dynamic_num_samples:
            clipped_ratio = total_trunc_masks.sum().item() / total_lengths.shape[0]
            self._metrics[mode]['completions/clipped_ratio'].append(clipped_ratio)

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                num_turns = torch.tensor(
                    gather_object([inp['rollout_infos']['num_turns'] for inp in inputs]), device=device)
                self._metrics[mode]['num_turns'].append(num_turns.float().mean().item())
        else:
            request_ids = gather_object([inp['request_id'] for inp in inputs])
            last_indices = self._get_last_indices(request_ids)

            final_trunc_masks = total_trunc_masks[last_indices]
            clipped_ratio = final_trunc_masks.sum().item() / final_trunc_masks.shape[0]
            self._metrics[mode]['completions/clipped_ratio'].append(clipped_ratio)

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                num_turns_all = torch.tensor(
                    gather_object([inp['rollout_infos']['num_turns'] for inp in inputs]), device=device)
                final_num_turns = num_turns_all[last_indices]
                self._metrics[mode]['num_turns'].append(final_num_turns.float().mean().item())

        return ga_batch_encoded_inputs

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model, return_outputs=True in mini-batch training
        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]
        if self.use_liger_loss:
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        mode = 'train' if self.model.training else 'eval'

        # Check batch size and decide processing strategy
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        should_chunk = self.dynamic_num_samples and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._compute_loss_single(model, inputs)
        else:
            # maybe dynamic rollout num for multi-turn training
            return self._compute_loss_chunked(model, inputs)

    def _compute_loss_single(self, model, inputs):
        """Original loss computation logic for single batch processing."""
        loss, metrics_data = self._compute_loss_and_metrics(model, inputs)
        self._update_metrics(metrics_data)
        return loss

    def _compute_loss_and_metrics(self, model, inputs):
        """Core loss computation without metrics recording."""
        mode = 'train' if self.model.training else 'eval'

        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        if self.template.padding_free:
            lengths = inputs['seq_lengths']
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, inputs, compute_entropy=self.compute_entropy)

        entropy_mask = None
        entropy_metrics = {}

        if self.compute_entropy:
            # fill the padded token with NaN
            entropies = entropies.masked_fill(completion_mask == 0, float('nan'))
            if self.args.log_entropy:
                if self.template.padding_free:
                    entropy_list = torch.split(entropies, lengths.tolist())
                    per_completion_entropies_mean = torch.stack([torch.nanmean(e) for e in entropy_list])
                else:
                    per_completion_entropies_mean = torch.nanmean(entropies, dim=1)
                global_per_completion_entropies_mean = gather(per_completion_entropies_mean)
                entropy_metrics = {
                    'entropy_logs': global_per_completion_entropies_mean.tolist(),
                    'entropy_mean': global_per_completion_entropies_mean.nanmean().item(),
                    'entropy_max': nanmax(global_per_completion_entropies_mean).item(),
                    'entropy_min': nanmin(global_per_completion_entropies_mean).item()
                }

            # compute the entropy threshold across all tokens in the batch
            if self.args.top_entropy_quantile < 1.0:
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics['entropy_threshold'] = entropy_threshold.item()
                entropy_mask = entropies >= entropy_threshold

        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong and truncated, '
                            'resulting in NaN some values for some metrics (e.g., KL)')
            if self.template.padding_free:
                truncated_mask = torch.repeat_interleave(truncated_mask, lengths).unsqueeze(0)
                assert truncated_mask.shape == completion_mask.shape
            else:
                truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        # Compute the KL divergence between the model and the reference model
        # Only compute KL for loss if kl_in_reward=False (GRPO style)
        if self.beta != 0.0 and not self.kl_in_reward:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        else:
            per_token_kl = None

        advantages = inputs['advantages']
        # When under on-policy training
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs['old_per_token_logps'] is None else inputs['old_per_token_logps'])

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            if self.template.padding_free:
                # split to batch, compute seq-level normalization
                log_ratio_list = torch.split(log_ratio.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())
                seq_weights = [(lr * m).sum() / m.sum().clamp(min=1.0) for lr, m in zip(log_ratio_list, mask_list)]
                seq_level_log_weights = torch.stack(seq_weights).to(log_ratio.dtype).unsqueeze(-1)
                if self.importance_sampling_level == 'sequence':
                    log_importance_weights = seq_level_log_weights
                else:
                    seq_level_log_weight = seq_level_log_weights.detach()
                    seq_level_log_weight = torch.repeat_interleave(seq_level_log_weight, lengths).unsqueeze(0)
                    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
            else:
                seq_level_log_weights = ((log_ratio * completion_mask).sum(-1)
                                         / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
                if self.importance_sampling_level == 'sequence':
                    log_importance_weights = seq_level_log_weights
                else:
                    # GSPO-token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
                    seq_level_log_weight = seq_level_log_weights.detach()
                    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight

        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'.")

        coef_1 = torch.exp(log_importance_weights)

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            if self.template.padding_free:
                advantages = advantages[-coef_1.shape[1]:]
                per_token_loss = -clamped_ratios * advantages.unsqueeze(0) * per_token_logps
            else:
                per_token_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            if self.template.padding_free:
                if self.importance_sampling_level == 'sequence':
                    # Expand sequence-level weights to token-level
                    coef_1 = torch.repeat_interleave(coef_1.squeeze(-1), lengths).unsqueeze(0)
                    coef_2 = torch.repeat_interleave(coef_2.squeeze(-1), lengths).unsqueeze(0)

                advantages = advantages[-coef_1.shape[1]:]
                per_token_loss1 = coef_1 * advantages.unsqueeze(0)
                per_token_loss2 = coef_2 * advantages.unsqueeze(0)
            else:
                per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == 'grpo':
            if self.template.padding_free:
                loss_list = torch.split(per_token_loss.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())
                sample_loss = [(loss * mask).sum() / mask.sum().clamp(min=1.0)
                               for loss, mask in zip(loss_list, mask_list)]
                loss = torch.stack(sample_loss).mean()
            else:
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            batch_size = lengths.shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        elif self.loss_type in ['cispo', 'dapo']:
            # CISPO and DAPO: Normalize by total completion tokens across all processes
            normalizer = inputs['num_items_in_batch'] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            # compute for token-level average
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Prepare metrics data
        metrics_data = {
            'mode': mode,
            'entropy': entropy_metrics,
            'completion_mask': completion_mask,
            'completion_token_count': completion_token_count,
        }

        if per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data['kl'] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        # Compute the clipped probability ratios
        if self.loss_type == 'cispo':
            # CISPO: Only track upper bound clipping
            if self.template.padding_free:
                is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(0) > 0)
            else:
                is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather_for_metrics(cispo_clip_ratio)
            metrics_data['clipping'] = {'cispo_clip_ratio': gathered_cispo_clip_ratio.nanmean().item()}
        else:
            if self.template.padding_free:
                is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(0) < 0)
                is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(0) > 0)
            else:
                is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
                is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
            gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
            gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)

            metrics_data['clipping'] = {
                'low_clip_mean': gathered_low_clip.nanmean().item(),
                'low_clip_min': nanmin(gathered_low_clip).item(),
                'high_clip_mean': gathered_high_clip.nanmean().item(),
                'high_clip_max': nanmax(gathered_high_clip).item(),
                'region_clip_mean': gathered_clip_ratio.nanmean().item()
            }
        if mode == 'train' and self.chord_sft_iterator is not None:
            loss = compute_chord_loss(self, grpo_loss=loss)

        return loss, metrics_data

    def _update_metrics(self, metrics_data):
        """Update metrics from metrics_data."""
        mode = metrics_data['mode']

        # Update entropy metrics
        if metrics_data['entropy']:
            entropy_metrics = metrics_data['entropy']
            if 'entropy_logs' in entropy_metrics:
                self._logs['entropy'].extend(entropy_metrics['entropy_logs'])
                self._metrics[mode]['entropy/mean'].append(entropy_metrics['entropy_mean'])
                self._metrics[mode]['entropy/max'].append(entropy_metrics['entropy_max'])
                self._metrics[mode]['entropy/min'].append(entropy_metrics['entropy_min'])
            if 'entropy_threshold' in entropy_metrics:
                self._metrics[mode]['entropy/threshold'].append(entropy_metrics['entropy_threshold'])

        # Update KL metrics
        if 'kl' in metrics_data:
            self._metrics[mode]['kl'].append(metrics_data['kl'])

        # Update clipping metrics
        if 'clipping' in metrics_data:
            clipping = metrics_data['clipping']
            if 'cispo_clip_ratio' in clipping:
                # CISPO
                self._metrics[mode]['cispo_clip_ratio'].append(clipping['cispo_clip_ratio'])
            else:
                self._metrics[mode]['clip_ratio/low_mean'].append(clipping['low_clip_mean'])
                self._metrics[mode]['clip_ratio/low_min'].append(clipping['low_clip_min'])
                self._metrics[mode]['clip_ratio/high_mean'].append(clipping['high_clip_mean'])
                self._metrics[mode]['clip_ratio/high_max'].append(clipping['high_clip_max'])
                self._metrics[mode]['clip_ratio/region_mean'].append(clipping['region_clip_mean'])

    def _compute_loss_chunked(self, model, inputs: DataType):
        """
        Compute loss in **fixed-size chunks** to reduce peak GPU memory.

        The function guarantees that **all ranks step through the same number of
        chunks**, so that collective communication remain synchronized
        even when local ``batch_size`` differs.
        """
        mode = 'train' if self.model.training else 'eval'
        chunk_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]

        # Decide how many chunks every rank must run
        batch_sizes = gather_object([batch_size])
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)

        # Re-compute chunk size so that max_chunks * new_chunk_size covers entire batch
        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        losses, weights = [], []
        all_metrics_data = []
        chunk_inputs = {}
        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)

            if start_idx < batch_size:
                chunk_inputs = self.get_chunked_inputs(inputs, start_idx, end_idx)

            # Compute loss and metrics for this chunk (without updating global metrics)
            chunk_loss, chunk_metrics_data = self._compute_loss_and_metrics(model, chunk_inputs)
            chunk_weight = end_idx - start_idx

            if start_idx < batch_size:
                losses.append(chunk_loss * chunk_weight)
                weights.append(chunk_weight)
                all_metrics_data.append((chunk_metrics_data, chunk_weight))

        # Compute weighted average loss
        total_weight = sum(weights)
        if total_weight > 0:
            final_loss = torch.stack(losses).sum() / total_weight
        else:
            final_loss = torch.tensor(0.0, device=model.device)

        # Aggregate metrics across all chunks
        self._aggregate_and_update_metrics(all_metrics_data, mode)

        return final_loss

    def _aggregate_and_update_metrics(self, all_metrics_data, mode):
        """Aggregate metrics from multiple chunks and update global metrics."""
        if not all_metrics_data:
            return

        # Separate metrics by type for aggregation
        entropy_logs, entropy_stats, kl_values = [], [], []
        clip_values = {'low': [], 'high': [], 'region': [], 'low_min': [], 'high_max': []}
        cispo_clip_values = []
        entropy_thresholds = []

        for chunk_metrics, chunk_weight in all_metrics_data:
            chunk_tokens = chunk_metrics['completion_token_count']

            # Collect entropy metrics
            if chunk_metrics['entropy']:
                entropy_metrics = chunk_metrics['entropy']
                if 'entropy_logs' in entropy_metrics:
                    entropy_logs.extend(entropy_metrics['entropy_logs'])
                    entropy_stats.append({
                        'mean': entropy_metrics['entropy_mean'],
                        'max': entropy_metrics['entropy_max'],
                        'min': entropy_metrics['entropy_min']
                    })
                if 'entropy_threshold' in entropy_metrics:
                    entropy_thresholds.append(entropy_metrics['entropy_threshold'])

            # Collect KL metrics
            if 'kl' in chunk_metrics:
                kl_values.append(chunk_metrics['kl'])

            # Collect clipping metrics (weighted by tokens)
            if 'clipping' in chunk_metrics:
                clipping = chunk_metrics['clipping']
                weight = chunk_tokens.item() if hasattr(chunk_tokens, 'item') else chunk_tokens
                if 'cispo_clip_ratio' in clipping:
                    cispo_clip_values.append((clipping['cispo_clip_ratio'], weight))
                else:
                    clip_values['low'].append((clipping['low_clip_mean'], weight))
                    clip_values['high'].append((clipping['high_clip_mean'], weight))
                    clip_values['region'].append((clipping['region_clip_mean'], weight))
                    clip_values['low_min'].append(clipping['low_clip_min'])
                    clip_values['high_max'].append(clipping['high_clip_max'])

        # Build aggregated metrics
        aggregated_metrics = {'mode': mode, 'entropy': {}}

        # Aggregate entropy
        if entropy_logs:
            # Directly update entropy logs
            self._logs['entropy'].extend(entropy_logs)
            aggregated_metrics['entropy'] = {
                'entropy_mean': sum(s['mean'] for s in entropy_stats) / len(entropy_stats),
                'entropy_max': max(s['max'] for s in entropy_stats),
                'entropy_min': min(s['min'] for s in entropy_stats)
            }
        if entropy_thresholds:
            aggregated_metrics['entropy']['entropy_threshold'] = sum(entropy_thresholds) / len(entropy_thresholds)

        # Aggregate KL
        if kl_values:
            aggregated_metrics['kl'] = sum(kl_values) / len(kl_values)

        # Aggregate clipping (token-weighted averages)
        def weighted_avg(values):
            return sum(v * w for v, w in values) / sum(w for _, w in values)

        if cispo_clip_values:
            # CISPO specific metric
            aggregated_metrics['clipping'] = {'cispo_clip_ratio': weighted_avg(cispo_clip_values)}
        elif clip_values['low']:
            # Two-sided clipping metrics
            aggregated_metrics['clipping'] = {
                'low_clip_mean': weighted_avg(clip_values['low']),
                'low_clip_min': min(clip_values['low_min']),
                'high_clip_mean': weighted_avg(clip_values['high']),
                'high_clip_max': max(clip_values['high_max']),
                'region_clip_mean': weighted_avg(clip_values['region'])
            }

        # Update metrics
        self._update_metrics(aggregated_metrics)

    def _get_per_token_logps_and_entropies_sp(
            self,
            model: torch.nn.Module,
            inputs: 'DataType',
            compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get per token logps for GRPO sequence parallel training"""
        try:
            from trl.trainer.utils import selective_log_softmax
        except ImportError:
            raise ImportError('trl is required for GRPO training. Please install it with: pip install trl')

        from swift.trainers.sequence_parallel.utils import GatherLoss
        from swift.trainers.sequence_parallel import sequence_parallel

        # original logits to keep
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        inputs = {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask', 'seq_lengths', 'num_items_in_batch'
            ]
        }
        sequence_parallel.prepare_inputs(inputs)
        with self._template_context(self.template):
            output = model(**inputs)
            logits = output.logits
        # split input_ids to labels
        position_ids = sequence_parallel.real_position_ids
        _, _, labels, _, _, _, _ = sequence_parallel.pad_and_split_inputs(
            None, None, input_ids.clone(), None, None, None, real_position_ids=position_ids)

        labels = torch.where(labels == -100, self.processing_class.pad_token_id, labels)
        logits = logits / self.temperature
        per_token_logps = selective_log_softmax(logits, labels)
        entropies = None
        per_token_logps, _ = GatherLoss.apply(per_token_logps, labels, 1, position_ids)
        if compute_entropy:
            entropies = entropy_from_logits(logits)
            entropies, _ = GatherLoss.apply(entropies, labels, 1, position_ids)

        per_token_logps = per_token_logps[:, -logits_to_keep - 1:-1]
        if compute_entropy:
            entropies = entropies[:, -logits_to_keep - 1:-1]
        # ignore the last token
        return per_token_logps, entropies

    @patch_profiling_decorator
    def _get_per_token_logps_and_entropies(self,
                                           model,
                                           inputs,
                                           compute_entropy=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log probabilities and entropies with memory-efficient batching.

        When rollout count is larger than expected, we process in smaller batches
        to control memory usage.
        """
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        mode = 'train' if self.model.training else 'eval'
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size  # noqa
        should_chunk = self.dynamic_num_samples and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._get_per_token_logps_and_entropies_single(model, inputs, compute_entropy=compute_entropy)
        else:
            return self._get_per_token_logps_and_entropies_chunked(model, inputs, compute_entropy=compute_entropy)

    def _get_per_token_logps_and_entropies_single(self,
                                                  model,
                                                  inputs,
                                                  compute_entropy=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.template.sequence_parallel_size > 1:
            return self._get_per_token_logps_and_entropies_sp(model, inputs, compute_entropy=compute_entropy)
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        use_local_entropy = not hasattr(super(), '_get_per_token_logps_and_entropies') and compute_entropy

        can_use_super = (not self.is_multimodal and 'logits_to_keep' in parameters and not use_local_entropy)
        if 'attention_mask' not in inputs:
            # when set padding_free true, the attention_mask is not in inputs
            can_use_super = False

        if can_use_super:
            # save memory
            if hasattr(super(), '_get_per_token_logps_and_entropies'):
                logps, entropies = super()._get_per_token_logps_and_entropies(
                    model, input_ids, inputs['attention_mask'], logits_to_keep, compute_entropy=compute_entropy)
            else:
                logps = super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
                entropies = None
        else:
            inputs = {
                k: v
                for k, v in inputs.items() if k not in [
                    'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                    'truncated_mask', 'seq_lengths', 'num_items_in_batch'
                ]
            }
            if 'logits_to_keep' in self.model_kwarg_keys:
                inputs['logits_to_keep'] = logits_to_keep + 1
            logits = model(**inputs).logits
            # exclude the last logit: it corresponds to the next token pred
            logits = logits[:, -(logits_to_keep + 1):-1, :]
            logits = logits / self.temperature
            input_ids = input_ids[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
            entropies = None
            if compute_entropy:
                entropies = entropy_from_logits(logits)

        return logps, entropies

    def _get_per_token_logps_and_entropies_chunked(self,
                                                   model,
                                                   inputs,
                                                   compute_entropy=False
                                                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log-probabilities (and optionally entropies) in **fixed-size
        chunks** to bound peak GPU memory.

        This routine **guarantees that every rank executes the same number of
        chunks**, even when the local batch sizes differ.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to compute log-probs and entropies.
        inputs : DataType
            A list of dictionary of tensors that constitute the full rollout/training batch.
        compute_entropy : bool, optional
            Whether to compute per-token entropies as well (default: False).

        Returns
        -------
        final_logps : torch.Tensor
            Concatenated per-token log-probabilities for the **entire batch**.
        final_entropies : torch.Tensor or None
            Concatenated per-token entropies, or ``None`` if ``compute_entropy`` is
            ``False``.
        """
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        mode = 'train' if self.model.training else 'eval'
        chunk_size = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size

        batch_sizes = gather_object([batch_size])  # list[int]
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)

        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        all_logps, all_entropies = [], [] if compute_entropy else None

        # Process in chunks
        chunk_inputs = {}
        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)

            if start_idx < end_idx:
                chunk_inputs = self.get_chunked_inputs(inputs, start_idx, end_idx)

            chunk_logps, chunk_entropies = self._get_per_token_logps_and_entropies_single(
                model, chunk_inputs, compute_entropy)

            if start_idx < end_idx:
                all_logps.append(chunk_logps)
                if compute_entropy and chunk_entropies is not None:
                    all_entropies.append(chunk_entropies)

        # Concatenate results
        final_logps = torch.cat(all_logps, dim=0)
        final_entropies = torch.cat(all_entropies, dim=0) if all_entropies else None

        return final_logps, final_entropies

    @patch_profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, inputs, logits_to_keep):
        # unwrap the model to access the model.model
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        if not self.is_multimodal:
            last_hidden_state = unwrapped_model.model(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        else:
            inputs = {
                k: v
                for k, v in inputs.items() if k not in [
                    'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                    'truncated_mask', 'seq_lengths', 'num_items_in_batch'
                ]
            }
            if 'logits_to_keep' in self.model_kwarg_keys:
                inputs['logits_to_keep'] = logits_to_keep + 1

            last_hidden_state = unwrapped_model.model(**inputs).last_hidden_state

        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        assert not self.template.padding_free
        assert self.advantage_estimator == 'grpo'
        input_ids = inputs['input_ids']
        logits_to_keep = inputs['logits_to_keep']
        completion_ids = input_ids[:, -logits_to_keep:]
        completion_mask = inputs['completion_mask']

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, inputs, logits_to_keep)
        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs['advantages'],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get('old_per_token_logps'),
            ref_per_token_logps=inputs.get('ref_per_token_logps'),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = 'eval' if self.control.should_evaluate else 'train'
        if self.beta != 0.0:
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]['clip_ratio'].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def evaluation_loop(self, dataloader, *args, **kwargs):
        # Wait for the training rollout to complete
        if self.args.async_generate:
            while not self.is_async_generate_train_rollout_done():
                time.sleep(0.1)
        if self._queue.empty() and self.args.async_generate:
            self._prefetch(dataloader)
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        self.eval_flag = True
        return output

    def training_step(self, model: nn.Module, inputs: DataType, num_items_in_batch=None) -> torch.Tensor:
        if self.args.async_generate:
            # Wait for the eval rollout to complete
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        return super().training_step(model, inputs, num_items_in_batch)

    def old_policy(self):
        if self.template.sequence_parallel_size == 1:
            return (self.num_iterations > 1
                    or self.args.gradient_accumulation_steps % self.args.steps_per_generation != 0)
        else:
            from swift.trainers.sequence_parallel import sequence_parallel
            return (self.num_iterations > 1 or self.args.gradient_accumulation_steps %
                    (self.args.steps_per_generation * sequence_parallel.world_size) != 0)

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
            if self.ref_model:
                self.offload_model(self.ref_model)
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
                if self.ref_model:
                    self.load_model(self.ref_model)
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()

    @patch_profiling_decorator
    def resample_encode_failed_inputs(self, inputs: DataType, n_try_fetch: int = 10) -> DataType:
        """
        Attempt to encode each input using the template. If encoding fails,
        resample from a backup iterator until successful or until the maximum
        number of retries is reached.

        Args:
            inputs (DataType): A list of input data samples, each containing a `messages` field.
            n_try_fetch (int, optional): Maximum number of retries to fetch a new sample
                when encoding fails. Defaults to 10.

        Returns:
            DataType: A list of successfully encoded input samples.

        Raises:
            RuntimeError: If encoding fails after `n_try_fetch` resampling attempts.
        """
        template = self.template
        last_messages = None
        last_valid_data = None

        for i, data in enumerate(inputs):
            # Skip samples with the same `messages` as the previous one.
            # If the last sample was successfully encoded, reuse it.
            if last_messages is not None and data['messages'] == last_messages:
                if last_valid_data is not None:
                    inputs[i] = last_valid_data
                    continue

            current_data = data
            n_try = 0

            while True:
                try:
                    # Attempt to encode the current sample.
                    remove_response(current_data['messages'])
                    template.encode(current_data)
                    # If successful, store the result and update the last valid data.
                    inputs[i] = current_data
                    last_messages = current_data['messages']
                    last_valid_data = current_data
                    break

                except Exception as e:
                    # Encoding failed — attempt to resample a new input.
                    logger.warning(f'Encoding failed for one sample; resampling a new input. {e}')
                    n_try += 1

                    # Stop if the maximum retry limit is exceeded.
                    if n_try > n_try_fetch:
                        raise RuntimeError('Failed to obtain a valid sample after multiple attempts. '
                                           'Consider increasing `max_length` or adjusting the '
                                           '`truncation_strategy` to avoid excessive truncation.')

                    # Fetch a new sample from the resampling iterator.
                    current_data = next(self.truncated_resample_iterator)[0]

        return inputs

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}

        logs.update(metrics)
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        # - entropy only includes samples that went through training (computed in _compute_loss)
        # - Other fields (e.g., prompt/completion/reward) are collected from rollout (in _prepare_inputs)
        # Therefore, if entropy exists, to ensure length consistency across fields,
        # we align all data based on the number of samples in entropy.
        seen_nums = len(self._logs['entropy']) \
            if 'entropy' in self._logs else len(self._logs['prompt'])
        if self.accelerator.is_main_process and self.log_completions:
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
                **{k: list(v)[:seen_nums]
                   for k, v in self._logs['rewards'].items()},
                'advantages': list(self._logs['advantages'])[:seen_nums],
            }
            for key, value in self._logs.items():
                if key not in table and key not in ['image', 'rewards']:
                    table[key] = list(value)[:seen_nums]

            if self.args.log_entropy:
                table.update({'entropy': list(self._logs['entropy'])[:seen_nums]})

            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None

            self.jsonl_writer.append(table)

            if report_to_wandb:
                import pandas as pd
                # Create a copy to avoid modifying the original table used by other loggers.
                wandb_table = table.copy()
                if self._logs.get('image'):
                    wandb_table['image'] = [
                        wandb.Image(load_pil_img(img)) if img is not None else None for img in self._logs['image']
                    ]
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = []
                    for header in headers:
                        row.append(table[header][i])
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})

    def is_async_generate_eval_rollout_done(self):
        return not self.eval_flag or not self.eval_queue.empty()

    def is_async_generate_train_rollout_done(self):
        return not self.train_queue.empty()

    def _gather_and_flatten(self, local_list, dtype=None, device=None, flatten_level: int = 1):
        """
        Gather data from all ranks with `gather_object` and flatten as required.

        Args
        ----
        local_list : Sequence[Any]
            The per-rank data to be gathered. Can be any picklable structure.
        dtype : torch.dtype, optional
            If provided, the flattened result is converted to a tensor with this dtype.
        device : torch.device, optional
            Target device for the resulting tensor. Ignored if dtype is None.
        flatten_level : int
            0  keep the outer list of per-rank results: List[rank_data]
            1  flatten across ranks: List[element]
            2  flatten one more level (assumes rank_data is iterable): List[sub_element]

        Returns
        -------
        Any
            - List[Any] when dtype is None
            - torch.Tensor when dtype is given
        """
        gathered = gather_object(local_list)  # List[rank][...] returned by gather_object

        if flatten_level == 0:
            flat = gathered
        elif flatten_level == 1:  # flatten over ranks
            flat = [elem for rank_data in gathered for elem in rank_data]
        elif flatten_level == 2:  # flatten one additional level
            flat = [item for rank_data in gathered for sublist in rank_data for item in sublist]
        else:
            raise ValueError(f'Invalid flatten_level: {flatten_level}')

        if dtype is not None:
            try:
                return torch.tensor(flat, dtype=dtype, device=device)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f'Cannot convert gathered+flattened data to tensor: {e}') from e
        return flat

    def _group_inputs_by_request_id(self, inputs: DataType) -> Dict[str, List[Dict]]:
        """
        Group inputs by request_id for multi-turn reward computation.

        Args:
            inputs: List of input dictionaries, each containing a 'request_id' field

        Returns:
            Dict[str, List[Dict]]: A dictionary where keys are request_ids and values are
                                  lists of input dictionaries with the same request_id
        """
        inputs_by_request_id = {}

        for input_data in inputs:
            request_id = input_data.get('request_id')
            if request_id is None:
                # Skip inputs without request_id
                continue

            if request_id not in inputs_by_request_id:
                inputs_by_request_id[request_id] = []

            inputs_by_request_id[request_id].append(input_data)

        return inputs_by_request_id

    def _get_trajectory_inputs(self, inputs: DataType) -> Dict[str, List[Dict]]:
        """
        Retrieve trajectory data corresponding to the request_ids present in the current inputs.

        This method performs the following steps:
        1. Extract the set of request_ids from the current inputs
        2. Gather all inputs across processes
        3. Filter out entries whose request_id is not present in the local inputs
        4. Group the remaining inputs by request_id
        5. Keep only trajectory data for request_ids found in the current inputs

        Args:
            inputs: The current batch of input data. Each item is a dictionary
                containing at least the field 'request_id'.

        Returns:
            Dict[str, List[Dict]]: A mapping from request_id to the list of
            corresponding input records (trajectory data).
        """
        # Collect request_id set from the current inputs
        current_request_ids = {input_data['request_id'] for input_data in inputs}

        # Gather all inputs across processes
        total_inputs = gather_object(inputs)

        # Keep only entries whose request_id exists in the current inputs
        filtered_total_inputs = [
            input_data for input_data in total_inputs if input_data['request_id'] in current_request_ids
        ]

        # Group inputs by request_id
        inputs_by_request_id = self._group_inputs_by_request_id(filtered_total_inputs)

        return inputs_by_request_id

    def _get_last_indices(self, request_ids: List[str]) -> torch.Tensor:
        seen = {}
        for i, rid in enumerate(request_ids):
            seen[rid] = i
        return torch.tensor(list(seen.values()), dtype=torch.long, device=self.accelerator.device)

    def get_chunked_inputs(self, inputs, start_idx, end_idx):
        chunk_inputs = {}
        # for LLM, slice the inputs
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                chunk_inputs[key] = val[start_idx:end_idx]
            else:
                chunk_inputs[key] = val
        if self.is_multimodal:
            # for MLLM, re-encode to get mm-related inputs
            origin_data = inputs['_origin_data'][start_idx:end_idx]
            template = self.template
            with self._template_context(template):
                encoded_data = [template.encode(data) for data in origin_data]
                chunk_inputs.update(to_device(template.data_collator(encoded_data), self.model.device))
                chunk_inputs.pop('labels', None)
        return chunk_inputs

    def _prepare_liger_loss(self):
        self.use_liger_loss = self.args.use_liger_kernel
        if self.use_liger_loss:
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
            kwargs = {}
            if 'importance_sampling_level' in inspect.signature(LigerFusedLinearGRPOLoss.__init__).parameters:
                kwargs['importance_sampling_level'] = self.importance_sampling_level
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
                **kwargs,
            )
            self._forward_redirection = _ForwardRedirection()

    def _prepare_metrics(self):
        args = self.args
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))
        self._logs = {
            'prompt': deque(maxlen=args.generation_batch_size),
            'completion': deque(maxlen=args.generation_batch_size),
            'rewards': defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            'advantages': deque(maxlen=args.generation_batch_size),
        }
        self.compute_entropy = self.args.log_entropy or self.top_entropy_quantile < 1.0
        if self.args.log_entropy:
            self._logs.update({'entropy': deque(maxlen=args.generation_batch_size)})

    def _collect_config_info(self) -> Dict[str, str]:
        config = {
            'dynamic_sample': str(self.dynamic_sample),
            'importance_sampling_level': str(self.importance_sampling_level),
            'advantage_estimator': str(self.advantage_estimator),
            'chord_sft_enabled': str(self.chord_sft_dataset is not None),
        }
        return config

    def _prepare_algorithm_params(self):
        args = self.args
        self.shuffle_dataset = args.dataset_shuffle

        self.loss_type = args.loss_type  # loss normalization
        self.scale_rewards = args.scale_rewards

        # GRPO, https://arxiv.org/abs/2402.03300
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper, Multi-step

        # DAPO, https://arxiv.org/abs/2503.14476
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.dynamic_sample = args.dynamic_sample
        self.max_resample_times = args.max_resample_times
        self.overlong_filter = args.overlong_filter

        # Entropy Mask, https://arxiv.org/abs/2506.01939
        self.top_entropy_quantile = args.top_entropy_quantile

        # GSPO, https://arxiv.org/abs/2507.18071
        self.importance_sampling_level = args.importance_sampling_level

        # RLOO,
        self.advantage_estimator = args.advantage_estimator
        self.kl_in_reward = args.kl_in_reward

    def _prepare_chord_dataset(self):
        # CHORD, https://arxiv.org/abs/2508.11408
        self.chord_sft_iterator = None
        if self.chord_sft_dataset:
            self.chord_sft_iterator = make_chord_sft_dataset(self, self.chord_sft_dataset)

    def _prepare_rewards(self, reward_funcs, reward_model=None, reward_templates=None):
        args = self.args
        device = self.accelerator.device

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if reward_funcs:
            for i, reward_func in enumerate(reward_funcs):
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif reward_func in prms:
                    reward_func_class = prms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.plugin')

        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        self.reward_model_plugins = [None] * len(self.reward_funcs)

        if reward_model is not None:
            reward_plugins = args.reward_model_plugin
            if reward_plugins is None:
                reward_plugins = ['default'] * len(reward_model)
            assert len(reward_plugins) == len(reward_model), (
                f"The number of 'reward_model_plugin' ({len(reward_plugins)}) does not match "
                f"the number of 'reward_model' ({len(reward_model)}). "
                "Please provide a corresponding 'reward_model_plugin' for each 'reward_model'.")
            for rm, rm_plugin, rm_template in zip(reward_model, reward_plugins, reward_templates):
                # Set encoding mode train(see details in Template.encode).
                # Set max_length to None to disable truncation, as the input length has already been truncated earlier.
                rm_template.set_mode('train')
                rm_template.max_length = None
                if rm_plugin not in rm_plugins:
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.llm.plugin')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(device)

        # after init trainer
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True)

    def _prepare_resample_data_iterator(self):

        def cyclic_iter(iterable):
            while True:
                for x in iterable:
                    yield x

        @contextmanager
        def seed_context():
            # Use a different seed to ensure the resample dataset does not overlap with train_dataset
            seed = self.args.seed
            self.args.seed = seed + 1
            yield
            self.args.seed = seed

        with seed_context():
            if self.args.dynamic_sample:
                self.dynamic_resample_iterator = cyclic_iter(self.get_train_dataloader())

            if self.template.truncation_strategy == 'raise':

                @contextmanager
                def single_sample_context():
                    # Patch generation-related parameters to ensure that only one sample is processed per iteration
                    # when resampling truncated data.
                    origin_ng = self.num_generations
                    origin_gbs = self.args.generation_batch_size
                    origin_spg = self.args.steps_per_generation
                    try:
                        self.num_generations = 1
                        self.args.generation_batch_size = 1
                        self.args.steps_per_generation = 1
                        yield
                    finally:
                        self.num_generations = origin_ng
                        self.args.generation_batch_size = origin_gbs
                        self.args.steps_per_generation = origin_spg

                with single_sample_context():
                    self.truncated_resample_iterator = cyclic_iter(self.get_train_dataloader())
