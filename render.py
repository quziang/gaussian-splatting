#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 导入必要的库和模块
# Import necessary libraries and modules
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# 尝试导入稀疏Adam优化器（可选依赖）
# Try to import sparse Adam optimizer (optional dependency)
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """
    渲染一组视图并保存渲染结果和真实图像
    Render a set of views and save rendered results and ground truth images
    
    Args:
        model_path: 模型路径 / Model path
        name: 数据集名称（如'train'或'test'） / Dataset name (e.g., 'train' or 'test')
        iteration: 迭代次数 / Iteration number
        views: 相机视图列表 / List of camera views
        gaussians: 高斯模型 / Gaussian model
        pipeline: 渲染管道参数 / Rendering pipeline parameters
        background: 背景颜色 / Background color
        train_test_exp: 训练测试曝光标志 / Train test exposure flag
        separate_sh: 分离球谐函数标志 / Separate spherical harmonics flag
    """
    # 创建渲染结果和真实图像的保存路径
    # Create save paths for rendered results and ground truth images
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 创建目录（如果不存在）
    # Create directories if they don't exist
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历每个视图进行渲染
    # Iterate through each view for rendering
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 渲染当前视图
        # Render current view
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        # 获取真实图像（前3个通道：RGB）
        # Get ground truth image (first 3 channels: RGB)
        gt = view.original_image[0:3, :, :]

        # 如果使用训练测试曝光，则裁剪图像的后半部分
        # If using train test exposure, crop the latter half of the image
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 保存渲染结果和真实图像
        # Save rendered result and ground truth image
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    """
    渲染训练集和测试集
    Render training and test sets
    
    Args:
        dataset: 数据集参数 / Dataset parameters
        iteration: 迭代次数 / Iteration number
        pipeline: 渲染管道参数 / Rendering pipeline parameters
        skip_train: 是否跳过训练集渲染 / Whether to skip training set rendering
        skip_test: 是否跳过测试集渲染 / Whether to skip test set rendering
        separate_sh: 分离球谐函数标志 / Separate spherical harmonics flag
    """
    with torch.no_grad():
        # 创建高斯模型和场景
        # Create Gaussian model and scene
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 设置背景颜色
        # Set background color
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 渲染训练集（如果未跳过）
        # Render training set if not skipped
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        # 渲染测试集（如果未跳过）
        # Render test set if not skipped
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # 设置命令行参数解析器
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)  # 模型参数 / Model parameters
    pipeline = PipelineParams(parser)  # 管道参数 / Pipeline parameters
    
    # 添加额外的命令行参数
    # Add additional command line arguments
    parser.add_argument("--iteration", default=-1, type=int)  # 指定迭代次数，-1表示最新 / Specify iteration, -1 for latest
    parser.add_argument("--skip_train", action="store_true")  # 跳过训练集渲染 / Skip training set rendering
    parser.add_argument("--skip_test", action="store_true")  # 跳过测试集渲染 / Skip test set rendering
    parser.add_argument("--quiet", action="store_true")  # 静默模式 / Quiet mode
    
    # 解析命令行参数
    # Parse command line arguments
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 初始化系统状态（随机数生成器）
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 执行渲染
    # Execute rendering
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)