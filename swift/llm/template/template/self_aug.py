import torch
import torch.nn.functional as F
import random
from typing import Tuple, List, Optional, Literal

# --- 导入的库 ---
import os
import argparse
import numpy as np
import decord
import imageio
import itertools
from tqdm import tqdm
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights
    from torchvision.utils import save_image
except ImportError:
    # 错误处理
    raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights, save_image = None, None, None, None, None
from torchvision.transforms.functional import gaussian_blur



def flow_guided_deformation(
    video_tensor: torch.Tensor,
    num_deformations: int = 3,
    strength: float = 20.0,
    falloff_radius_ratio: float = 0.1,
    duration: int = 4,
    region: Literal['any', 'static', 'motion'] = 'motion',
    flow_path = "",
    flow_index = None,
    return_aug_info=False
):
    """
    使用光流指导的位移场来制造平滑、无边界的局部微形变。
    可以选择在静止或运动区域施加扰动。
    
    (已修复 torchvision 版本兼容性问题)

    Args:
        ...
        flow_model_instance: 预加载的光流模型实例.
        flow_weights_enum: 预加载的权重枚举对象 (e.g., Raft_Small_Weights.DEFAULT).
    """
    perturbed_video = video_tensor.clone()
    T, C, H, W = video_tensor.shape
    device = video_tensor.device

    random.shuffle(flow_index)
    for i in range(num_deformations):
        if T < max(duration, 2):
            continue
            
        # start_frame = random.randint(0, T - max(duration, 2))
        start_frame = flow_index[i]
        
        center_y, center_x = None, None
        
        if region in ['static', 'motion']:
            flow_path_cur = flow_path.replace(".mp4", f"_flow_{start_frame}.pt")
            flow_fw = torch.load(flow_path_cur).to(device)
            motion_map = torch.sqrt(flow_fw[0]**2 + flow_fw[1]**2).view(-1)
            if region == 'motion': sampling_weights = motion_map
            else: sampling_weights = 1.0 / (motion_map + 1e-6)
            if sampling_weights.sum() > 0:
                pixel_index = torch.multinomial(sampling_weights, 1).item()
                center_y, center_x = pixel_index // W, pixel_index % W

        if center_y is None or center_x is None:
            center_y, center_x = random.uniform(0, H-1), random.uniform(0, W-1)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        radius = min(H, W) * falloff_radius_ratio
        dist_sq = (grid_y - center_y)**2 + (grid_x - center_x)**2
        gaussian_weights = torch.exp(-dist_sq / (radius**2))
        angle = random.uniform(0, 2 * 3.14159)
        angle = torch.tensor(angle)
        direction_vec = torch.tensor([torch.cos(angle), torch.sin(angle)], device=device)
        displacement_field = strength * gaussian_weights.unsqueeze(-1) * direction_vec.view(1, 1, 2)
        original_grid = torch.stack((grid_x, grid_y), dim=-1)
        new_grid = original_grid + displacement_field
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
        frames_to_warp = video_tensor[start_frame : start_frame + duration]
        warped_frames = F.grid_sample(
            frames_to_warp, new_grid.unsqueeze(0).repeat(duration, 1, 1, 1),
            mode='bilinear', padding_mode='border', align_corners=False
        )
        perturbed_video[start_frame : start_frame + duration] = warped_frames
        
    if return_aug_info:
        x1, x2 = min(center_x - radius, 0), max(center_x + radius, W)
        y1, y2 = min(center_y - radius, 0), max(center_y + radius, H)
        fps = float(os.environ.get("FPS"))
        return perturbed_video, ((start_frame/fps, (start_frame+duration)/fps), (int(x1*1000/W), int(y1*1000/H), int(x2*1000/W), int(y2*1000/H)))
    return perturbed_video


def motion_perturbation(
    video_tensor: torch.Tensor,
    strength: float = 0.5,
    falloff_radius_ratio: float = 0.15, # 使用高斯衰减半径
    num_perturbations: int = 3,
    flow_path = "",
    flow_index = None,
    return_aug_info=False
):
    """
    通过平滑扰动高质量光流场，制造更清晰、更逼真的微小运动不一致性。

    优化点:
    1. 使用高斯衰减的平滑扰动场，避免硬边界和撕裂。
    2. 可选使用更高质量的 raft_large 模型。
    3. 可选进行前向-后向一致性检查，只在光流预测可靠的区域施加扰动。
    """
    if raft_small is None:
        print("Optical flow model not available. Skipping perturbation.")
        return video_tensor

    perturbed_video = video_tensor.clone()
    T, C, H, W = video_tensor.shape
    device = video_tensor.device

    random.shuffle(flow_index)
    for i in range(num_perturbations):
        if T < 2:
            continue
        
        # t = random.randint(0, T - 2)
        t = flow_index[i]
        frame1, frame2 = video_tensor[t], video_tensor[t+1]

        flow_path_cur = flow_path.replace(".mp4", f"_flow_{t}.pt")
        flow_fw = torch.load(flow_path_cur).to(device)
        
        # --- 优化2：平滑的扰动场 ---
        # 随机选择扰动中心
        center_y, center_x = random.uniform(0, H-1), random.uniform(0, W-1)
        
        # 创建高斯权重
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        radius = min(H, W) * falloff_radius_ratio
        dist_sq = (grid_y - center_y)**2 + (grid_x - center_x)**2
        gaussian_weights = torch.exp(-dist_sq / (radius**2)) # [H, W]

        # 生成随机方向的扰动，并应用高斯权重
        angle = random.uniform(0, 2 * 3.14159)
        angle = torch.tensor(angle)
        direction_vec = torch.tensor([torch.cos(angle), torch.sin(angle)], device=device).view(2, 1, 1)
        
        # [2, H, W]
        smooth_perturbation = strength * gaussian_weights.unsqueeze(0) * direction_vec
        
        perturbed_flow = flow_fw + smooth_perturbation

        # --- 后续的 Warping 过程 ---
        # (这部分与原函数类似，但使用的是 perturbed_flow)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        flow_permuted = perturbed_flow.permute(1, 2, 0)
        new_grid = torch.stack((grid_x + flow_permuted[..., 0], grid_y + flow_permuted[..., 1]), dim=-1)
        
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
        
        warped_frame2 = F.grid_sample(
            frame1.unsqueeze(0), 
            new_grid.unsqueeze(0), 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )
        
        perturbed_video[t+1] = warped_frame2.squeeze(0)

    return perturbed_video




def fade_perturbation(
    video_tensor: torch.Tensor,
    num_events: int = 3,
    patch_size_ratio_range: Tuple[float, float] = (0.05, 0.1),
    fade_duration: int = 8,
    shape_modes: List[str] = ['ellipse'],       # ['rectangle', 'ellipse', 'blob'],
    background_modes: List[str] =  ['temporal', 'blur'],
    return_aug_info=False     # ['average', 'temporal', 'blur']
):
    """
    在视频中随机选择区域，以不规则形状和多样化背景，使其平滑地消失或出现。

    Args:
        video_tensor (torch.Tensor): 输入视频张量, shape [T, C, H, W], 范围 [0, 1].
        num_events (int): 发生渐隐/渐现事件的次数.
        patch_size_ratio_range (Tuple[float, float]): 区域尺寸占图像比例的随机范围.
        fade_duration (int): 渐变持续的帧数.
        shape_modes (List[str]): 扰动区域的形状选项: 'rectangle', 'ellipse', 'blob'.
        background_modes (List[str]): 渐变背景的类型: 'average', 'temporal', 'blur'.

    Returns:
        torch.Tensor: 扰动后的视频张量, shape [T, C, H, W].
    """
    perturbed_video = video_tensor.clone()
    T, C, H, W = video_tensor.shape
    device = video_tensor.device

    if not shape_modes:
        print("Warning: No valid shape modes available. Skipping perturbation.")
        return perturbed_video

    for _ in range(num_events):
        if T < fade_duration + 1:  # +1 用于时间错位背景
            continue

        # 1. --- 确定扰动参数 ---
        start_frame = random.randint(1, T - fade_duration) # 从第1帧开始，为temporal背景留出空间
        patch_size_ratio = random.uniform(*patch_size_ratio_range)
        patch_h, patch_w = int(H * patch_size_ratio), int(W * patch_size_ratio)
        
        # 保证patch大小不为0
        if patch_h == 0 or patch_w == 0: continue

        h_start = random.randint(0, H - patch_h)
        w_start = random.randint(0, W - patch_w)
        h_end, w_end = h_start + patch_h, w_start + patch_w
        
        # 2. --- 创建不规则形状的蒙版 (Mask) ---
        shape = random.choice(shape_modes)
        mask = torch.zeros((patch_h, patch_w), device=device)

        if shape == 'ellipse':
            center_x, center_y = patch_w / 2, patch_h / 2
            axis_x, axis_y = random.uniform(0.3, 1.0) * center_x, random.uniform(0.3, 1.0) * center_y
            angle = random.uniform(0, 2 * np.pi)
            
            y, x = torch.meshgrid(torch.arange(patch_h, device=device), torch.arange(patch_w, device=device), indexing='ij')
            x, y = x - center_x, y - center_y
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = (x * cos_a + y * sin_a)
            y_rot = (-x * sin_a + y * cos_a)
            
            mask_condition = (x_rot**2 / axis_x**2 + y_rot**2 / axis_y**2) <= 1
            mask[mask_condition] = 1.0
        
        else: # 'rectangle' 或 'blob' 备选方案
            mask.fill_(1.0)

        # 增加一个平滑的边界，防止边缘过于锐利
        mask = gaussian_blur(mask.unsqueeze(0).unsqueeze(0), kernel_size=11, sigma=3).squeeze()
        mask = (mask - mask.min()) / (mask.max() - mask.min()) # 重新归一化到[0,1]
        mask = mask.view(1, 1, patch_h, patch_w) # 准备好广播

        # 3. --- 创建多样化的背景 ---
        patch_content = perturbed_video[start_frame : start_frame + fade_duration, :, h_start:h_end, w_start:w_end]
        background_type = random.choice(background_modes)
        
        if background_type == 'temporal':
            # 使用前一帧的内容作为背景
            background = video_tensor[start_frame - 1, :, h_start:h_end, w_start:w_end].unsqueeze(0).expand_as(patch_content)
        elif background_type == 'blur':
            # 使用模糊后的内容作为背景
            # 逐帧模糊以避免GPU显存问题
            blurred_frames = [gaussian_blur(patch_content[i].unsqueeze(0), kernel_size=21, sigma=10) for i in range(fade_duration)]
            background = torch.cat(blurred_frames, dim=0)
        else: # 'average'
            # 使用区域平均色作为背景
            background = patch_content.mean(dim=[0, 2, 3], keepdim=True).expand_as(patch_content)

        # 4. --- 应用渐变混合 ---
        # is_fade_out = random.choice([True, False])
        is_fade_out = True
        alphas = torch.linspace(1, 0, fade_duration, device=device) if is_fade_out else torch.linspace(0, 1, fade_duration, device=device)
        alphas = alphas.view(fade_duration, 1, 1, 1) # [duration, 1, 1, 1]

        # 计算alpha混合后的内容
        blended_content = alphas * patch_content + (1 - alphas) * background
        
        # 使用不规则蒙版将混合内容应用到原始块上
        final_patch = patch_content * (1 - mask) + blended_content * mask
        
        perturbed_video[start_frame : start_frame + fade_duration, :, h_start:h_end, w_start:w_end] = final_patch
        
    if return_aug_info:
        fps = float(os.environ.get("FPS"))
        return perturbed_video, ((start_frame/fps, (start_frame+fade_duration)/fps), (int(w_start*1000/W), int(h_start*1000/H), int(w_end*1000/W), int(h_end*1000/H)))
    return perturbed_video



def local_temporal_shuffle(
    video_tensor: torch.Tensor,
    num_shuffled_chunks: int = 3, # 要打乱的块的数量
    chunk_size: int = 4,          # 每个块的大小（帧数）
    return_aug_info=False
):
    """
    在一个视频中随机选择 n 个不重叠的时间块，并只在这些块内部打乱帧的顺序。

    Args:
        video_tensor (torch.Tensor): 输入视频张量, shape [T, C, H, W].
        num_shuffled_chunks (int): 要随机选择并打乱的块的数量。
        chunk_size (int): 每个时间块的大小（帧数）。

    Returns:
        torch.Tensor: 扰动后的视频张量。
    """
    perturbed_video = video_tensor.clone()
    T = video_tensor.shape[0]

    if chunk_size <= 1:
        return perturbed_video # 无法在大小为1的块内shuffle

    # 1. 计算所有可能的不重叠块的起始索引
    possible_starts = list(range(0, T - chunk_size + 1, chunk_size))
    
    if not possible_starts:
        return perturbed_video # 视频太短，无法形成一个完整的块

    # 2. 随机选择 num_shuffled_chunks 个块来进行 shuffle
    # 如果可能的块数少于指定的数量，则选择所有可能的块
    num_to_select = min(num_shuffled_chunks, len(possible_starts))
    chunks_to_shuffle_starts = random.sample(possible_starts, num_to_select)

    # 3. 对选定的每个块进行内部 shuffle
    for start_idx in chunks_to_shuffle_starts:
        end_idx = start_idx + chunk_size
        
        # 提取块
        chunk = perturbed_video[start_idx:end_idx]
        
        # 生成随机顺序并应用
        indices = torch.randperm(chunk_size)
        shuffled_chunk = chunk[indices]
        
        # 将打乱后的块放回
        perturbed_video[start_idx:end_idx] = shuffled_chunk
        
    return perturbed_video


def local_spatiotemporal_flicker(
    video_tensor: torch.Tensor,
    num_flickers: int = 10,
    cube_size: Tuple[int, int, int] = (4, 64, 64),
    return_aug_info=False
):
    """
    对视频施加固定数量的微小时空块闪烁扰动。
    随机选择N个时空小方块，并将它们替换为相邻时间帧的对应方块。

    Args:
        video_tensor (torch.Tensor): 输入视频张量, shape [T, C, H, W], 范围 [0, 1].
        num_flickers (int): 要替换的时空块的数量 (N).
        cube_size (Tuple[int, int, int]): 时空块的大小 (depth, height, width).

    Returns:
        torch.Tensor: 扰动后的视频张量, shape [T, C, H, W].
    """
    perturbed_video = video_tensor.clone()
    T, C, H, W = video_tensor.shape
    d, h, w = cube_size

    # 1. 计算视频可以被分割成多少个时空块
    num_cubes_t = T // d
    num_cubes_h = H // h
    num_cubes_w = W // w

    # 如果块大小不合适，无法分割，则直接返回原视频
    if num_cubes_t == 0 or num_cubes_h == 0 or num_cubes_w == 0:
        print("Warning: cube_size is too large for the video tensor. No perturbation applied.")
        return perturbed_video

    # 2. 生成所有可能的时空块的索引 (t_idx, h_idx, w_idx)
    all_possible_cube_indices = list(itertools.product(
        range(num_cubes_t),
        range(num_cubes_h),
        range(num_cubes_w)
    ))

    # 3. 从所有可能的块中随机选择 N 个进行替换
    # 如果 N 大于总块数，则替换所有块
    num_total_cubes = len(all_possible_cube_indices)
    num_to_replace = min(num_flickers, num_total_cubes)
    
    # random.sample确保选出的索引是唯一的（不重复）
    selected_indices = random.sample(all_possible_cube_indices, k=num_to_replace)

    # 4. 遍历被选中的 N 个块并执行替换操作
    for t_idx, h_idx, w_idx in selected_indices:
        # 计算当前块的起始和结束坐标
        t_start, t_end = t_idx * d, (t_idx + 1) * d
        h_start, h_end = h_idx * h, (h_idx + 1) * h
        w_start, w_end = w_idx * w, (w_idx + 1) * w

        # 随机选择时间偏移（向前或向后一帧）
        # 这里我们选择偏移一个基本时间单位，而不是整个块的深度
        time_shift = random.choice([-1, 1]) 
        
        # 计算源块的时间坐标
        src_t_start, src_t_end = t_start + time_shift, t_end + time_shift

        # 确保源块的坐标在视频的时间范围内
        if 0 <= src_t_start and src_t_end <= T:
            # 从原始视频中提取源块
            source_cube = video_tensor[src_t_start:src_t_end, :, h_start:h_end, w_start:w_end]
            # 将目标块替换为源块
            perturbed_video[t_start:t_end, :, h_start:h_end, w_start:w_end] = source_cube
            
    if return_aug_info:
        fps = float(os.environ.get("FPS"))
        return perturbed_video, ((t_start/fps, t_end/fps), (int(w_start*1000/W), int(h_start*1000/H), int(w_end*1000/W), int(h_end*1000/H)))
    return perturbed_video



# def flow_guided_deformation(
#     video_tensor: torch.Tensor,
#     num_deformations: int = 3,
#     strength: float = 20.0,
#     falloff_radius_ratio: float = 0.1,
#     duration: int = 4,
#     region: Literal['any', 'static', 'motion'] = 'motion',
#     # 传入预加载的模型实例和其对应的权重对象
#     flow_model_instance: torch.nn.Module = None, 
#     flow_weights_enum = None, # -> 传入 Raft_Small_Weights.DEFAULT
# ) -> torch.Tensor:
#     """
#     使用光流指导的位移场来制造平滑、无边界的局部微形变。
#     可以选择在静止或运动区域施加扰动。
    
#     (已修复 torchvision 版本兼容性问题)

#     Args:
#         ...
#         flow_model_instance: 预加载的光流模型实例.
#         flow_weights_enum: 预加载的权重枚举对象 (e.g., Raft_Small_Weights.DEFAULT).
#     """
#     perturbed_video = video_tensor.clone()
#     T, C, H, W = video_tensor.shape
#     device = video_tensor.device

#     for i in range(num_deformations):
#         if T < max(duration, 2):
#             continue
            
#         start_frame = random.randint(0, T - max(duration, 2))
        
#         center_y, center_x = None, None
        
#         if region in ['static', 'motion'] and raft_small is not None:
#             # ... (模型加载逻辑保持不变) ...
#             local_model = flow_model_instance
#             local_weights = flow_weights_enum
#             if local_model is None:
#                 if local_weights is None:
#                     local_weights = Raft_Small_Weights.DEFAULT
#                 local_model = raft_small(weights=local_weights).to(device)
#                 local_model.eval()
            
#             transforms = local_weights.transforms()
#             frame1, frame2 = video_tensor[start_frame], video_tensor[start_frame + 1]
#             # frame1_uint8 = (frame1 * 255).to(torch.uint8).unsqueeze(0)
#             # frame2_uint8 = (frame2 * 255).to(torch.uint8).unsqueeze(0)
#             # img1_batch, img2_batch = transforms(frame1_uint8, frame2_uint8)
#             img1_batch, img2_batch = transforms(frame1.unsqueeze(0), frame2.unsqueeze(0))

#             with torch.no_grad():
#                 predicted_flows = local_model(img1_batch.to(device), img2_batch.to(device))
#                 flow = predicted_flows[-1][0] # Shape: [2, H, W]

#             motion_map = torch.sqrt(flow[0]**2 + flow[1]**2).view(-1)
#             if region == 'motion': sampling_weights = motion_map
#             else: sampling_weights = 1.0 / (motion_map + 1e-6)
#             if sampling_weights.sum() > 0:
#                 pixel_index = torch.multinomial(sampling_weights, 1).item()
#                 center_y, center_x = pixel_index // W, pixel_index % W

#         if center_y is None or center_x is None:
#             center_y, center_x = random.uniform(0, H-1), random.uniform(0, W-1)

#         # ... (后续的形变代码保持不变) ...
#         grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
#         radius = min(H, W) * falloff_radius_ratio
#         dist_sq = (grid_y - center_y)**2 + (grid_x - center_x)**2
#         gaussian_weights = torch.exp(-dist_sq / (radius**2))
#         angle = random.uniform(0, 2 * 3.14159)
#         angle = torch.tensor(angle)
#         direction_vec = torch.tensor([torch.cos(angle), torch.sin(angle)], device=device)
#         displacement_field = strength * gaussian_weights.unsqueeze(-1) * direction_vec.view(1, 1, 2)
#         original_grid = torch.stack((grid_x, grid_y), dim=-1)
#         new_grid = original_grid + displacement_field
#         new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
#         new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
#         frames_to_warp = video_tensor[start_frame : start_frame + duration]
#         warped_frames = F.grid_sample(
#             frames_to_warp, new_grid.unsqueeze(0).repeat(duration, 1, 1, 1),
#             mode='bilinear', padding_mode='border', align_corners=False
#         )
#         perturbed_video[start_frame : start_frame + duration] = warped_frames
        
#     return perturbed_video


# def motion_perturbation(
#     video_tensor: torch.Tensor,
#     strength: float = 0.5,
#     falloff_radius_ratio: float = 0.15, # 使用高斯衰减半径
#     num_perturbations: int = 3,
#     use_large_model: bool = False, # 可选更高质量模型
#     consistency_check: bool = True, # 可选前向-后向一致性检查
#     consistency_thresh: float = 1.0, # 一致性检查的阈值（像素）
#     # 传入预加载的模型实例
#     flow_model_instance=None, 
#     flow_weights_enum=None
# ) -> torch.Tensor:
#     """
#     通过平滑扰动高质量光流场，制造更清晰、更逼真的微小运动不一致性。

#     优化点:
#     1. 使用高斯衰减的平滑扰动场，避免硬边界和撕裂。
#     2. 可选使用更高质量的 raft_large 模型。
#     3. 可选进行前向-后向一致性检查，只在光流预测可靠的区域施加扰动。
#     """
#     if raft_small is None:
#         print("Optical flow model not available. Skipping perturbation.")
#         return video_tensor

#     perturbed_video = video_tensor.clone()
#     T, C, H, W = video_tensor.shape
#     device = video_tensor.device

#     # --- 模型加载（优化） ---
#     local_model = flow_model_instance
#     local_weights = flow_weights_enum
#     if local_model is None:
#         if use_large_model and raft_large is not None:
#             local_weights = Raft_Large_Weights.DEFAULT if local_weights is None else local_weights
#             local_model = raft_large(weights=local_weights).to(device)
#         else:
#             local_weights = Raft_Small_Weights.DEFAULT if local_weights is None else local_weights
#             local_model = raft_small(weights=local_weights).to(device)
#         local_model.eval()
    
#     transforms = local_weights.transforms()

#     for _ in range(num_perturbations):
#         if T < 2:
#             continue
        
#         t = random.randint(0, T - 2)
#         frame1, frame2 = video_tensor[t], video_tensor[t+1]
        
#         # 预处理
#         img1_batch, img2_batch = transforms(frame1.unsqueeze(0), frame2.unsqueeze(0))
        
#         with torch.no_grad():
#             # 1. 计算前向光流 (frame1 -> frame2)
#             predicted_flows_fw = local_model(img1_batch, img2_batch)
#             flow_fw = predicted_flows_fw[-1][0] # [2, H, W]

#             reliable_mask = torch.ones(1, H, W, device=device, dtype=torch.bool)

#             if consistency_check:
#                 # 2. 计算后向光流 (frame2 -> frame1)
#                 predicted_flows_bw = local_model(img2_batch, img1_batch)
#                 flow_bw = predicted_flows_bw[-1][0]
                
#                 # 3. 进行一致性检查
#                 # 将后向光流 warp 到前向光流的坐标系
#                 grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
#                 original_grid = torch.stack((grid_x, grid_y), dim=-1).to(device) # [H, W, 2]
                
#                 warped_grid = original_grid + flow_fw.permute(1, 2, 0)
#                 warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
#                 warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0

#                 # 从后向光流场中采样
#                 warped_flow_bw = F.grid_sample(flow_bw.unsqueeze(0), warped_grid.unsqueeze(0), mode='bilinear', align_corners=False).squeeze(0)
                
#                 # 计算差异：flow_fw + warped_flow_bw 应该接近于0
#                 diff = torch.sqrt((flow_fw[0] + warped_flow_bw[0])**2 + (flow_fw[1] + warped_flow_bw[1])**2)
                
#                 # 标记可靠区域
#                 reliable_mask = (diff < consistency_thresh).unsqueeze(0) # [1, H, W]
        
#         # --- 优化2：平滑的扰动场 ---
#         # 随机选择扰动中心
#         center_y, center_x = random.uniform(0, H-1), random.uniform(0, W-1)
        
#         # 创建高斯权重
#         grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
#         radius = min(H, W) * falloff_radius_ratio
#         dist_sq = (grid_y - center_y)**2 + (grid_x - center_x)**2
#         gaussian_weights = torch.exp(-dist_sq / (radius**2)) # [H, W]

#         # 生成随机方向的扰动，并应用高斯权重
#         angle = random.uniform(0, 2 * 3.14159)
#         angle = torch.tensor(angle)
#         direction_vec = torch.tensor([torch.cos(angle), torch.sin(angle)], device=device).view(2, 1, 1)
        
#         # [2, H, W]
#         smooth_perturbation = strength * gaussian_weights.unsqueeze(0) * direction_vec
        
#         # 只在可靠区域施加扰动
#         final_perturbation = smooth_perturbation * reliable_mask
        
#         # 将平滑扰动应用到原始光流上
#         perturbed_flow = flow_fw + final_perturbation
#         # perturbed_flow = flow_fw + smooth_perturbation

#         # --- 后续的 Warping 过程 ---
#         # (这部分与原函数类似，但使用的是 perturbed_flow)
#         grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
#         flow_permuted = perturbed_flow.permute(1, 2, 0)
#         new_grid = torch.stack((grid_x + flow_permuted[..., 0], grid_y + flow_permuted[..., 1]), dim=-1)
        
#         new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
#         new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
        
#         warped_frame2 = F.grid_sample(
#             frame1.unsqueeze(0), 
#             new_grid.unsqueeze(0), 
#             mode='bilinear', 
#             padding_mode='border', 
#             align_corners=False
#         )
        
#         perturbed_video[t+1] = warped_frame2.squeeze(0)

#     return perturbed_video






# # ==========================================================================================
# # --- 视频处理主逻辑 ---
# # ==========================================================================================

# def save_tensor_as_video(video_tensor: torch.Tensor, output_path: str, fps: float):
#     """辅助函数：将torch张量保存为视频文件。"""
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # 将张量移回CPU，转换格式为 [T, H, W, C] uint8
#     output_frames = (video_tensor.cpu().permute(0, 2, 3, 1) * 255.0).clip(0, 255).to(torch.uint8).numpy()

#     print(f"Saving video to {output_path} with FPS={fps:.2f}...")
#     try:
#         imageio.mimwrite(output_path, output_frames, fps=fps, codec='libx264', quality=8)
#         print(f"Successfully saved {output_path}")
#     except Exception as e:
#         print(f"Error saving video file: {e}")
#         print("You might need to install 'ffmpeg'. Try: conda install ffmpeg -c conda-forge")

# def process_video(
#     input_path: str,
#     output_dir: str,
#     perturbation_type: str,
#     device: str,
#     target_fps: Optional[int] = None,
#     resize: Optional[Tuple[int, int]] = None
# ):
#     """
#     读取视频，进行预处理（重采样/缩放），应用指定的扰动，并保存预处理后和扰动后的两个视频。
#     """
#     print("--- Starting video processing ---")
#     print(f"Input video: {input_path}")
    
#     # 1. 构造输出文件名
#     base_name = os.path.basename(input_path)
#     file_name_no_ext, file_ext = os.path.splitext(base_name)
    
#     preprocessed_output_path = os.path.join(output_dir, base_name)
#     perturbed_output_path = os.path.join(output_dir, f"{file_name_no_ext}_{perturbation_type}{file_ext}")

#     print(f"Preprocessed video will be saved to: {preprocessed_output_path}")
#     print(f"Perturbed video will be saved to: {perturbed_output_path}")

#     try:
#         # vr = decord.VideoReader(input_path, height=h, width=w)
#         vr = decord.VideoReader(input_path)
#         source_fps = vr.get_avg_fps()
#         total_frames = len(vr)
#         print(f"Successfully opened video. Original specs: {total_frames} frames, {source_fps:.2f} FPS.")
#     except Exception as e:
#         print(f"Error opening video file with decord: {e}")
#         return

#     # 3. 计算需要采样的帧索引
#     display_fps = target_fps if target_fps is not None else source_fps
    
#     if target_fps is not None and target_fps != source_fps:
#         print(f"Resampling FPS from {source_fps:.2f} to {target_fps}.")
#         num_output_frames = int(total_frames * (target_fps / source_fps))
#         frame_indices = np.linspace(0, total_frames - 1, num_output_frames, dtype=int)
#     else:
#         print("Reading video at its original FPS.")
#         frame_indices = np.arange(total_frames, dtype=int)

#     # 4. 批量读取帧并转换为 PyTorch 张量
#     print(f"Reading and preprocessing {len(frame_indices)} frames...")
#     frames_np = vr.get_batch(frame_indices).asnumpy()
    
#     # [T, H, W, C] -> [T, C, H, W], 归一化到 [0, 1]
#     video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
#     video_tensor = video_tensor.to(device)

#     print(f"Video tensor created with shape: {video_tensor.shape}")
#     h, w = (None, None) if resize is None else (resize[0], resize[1])
#     if resize:
#         print(f"Resizing video to {h}x{w}.")
#         video = transforms.functional.resize(
#             video,
#             [h, w],
#             interpolation=InterpolationMode.BICUBIC,
#             antialias=True,
#         ).float()

#     # 5. 保存预处理后的视频
#     save_tensor_as_video(video_tensor, preprocessed_output_path, display_fps)

#     # 6. 应用选择的扰动
#     print(f"Applying '{perturbation_type}' perturbation...")
#     perturbation_fn = {
#         'flicker': subtle_cube_flicker,
#         'deformation': guided_flow_like_deformation,     # localized_micro_deformation,
#         'fade': content_aware_disappearance,    # fade_in_out_perturbation,
#         'advanced_fade': advanced_fade_perturbation,
#         'flow': refined_optical_flow_perturbation    # optical_flow_micro_perturbation
#     }.get(perturbation_type)

#     if perturbation_fn:
#         print(type(video_tensor))
#         perturbed_tensor = perturbation_fn(video_tensor)
#         print("Perturbation applied.")
#     else:
#         print(f"Error: Unknown perturbation type '{perturbation_type}'")
#         return

#     # 7. 保存扰动后的视频
#     save_tensor_as_video(perturbed_tensor, perturbed_output_path, display_fps)
    
#     print("--- Video processing finished ---")
