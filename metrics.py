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
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    """
    读取渲染图像和真实图像
    Read rendered images and ground truth images
    
    Args:
        renders_dir: 渲染图像目录路径 / Directory path for rendered images
        gt_dir: 真实图像目录路径 / Directory path for ground truth images
    
    Returns:
        renders: 渲染图像张量列表 / List of rendered image tensors
        gts: 真实图像张量列表 / List of ground truth image tensors
        image_names: 图像文件名列表 / List of image file names
    """
    renders = []
    gts = []
    image_names = []
    
    # 遍历渲染目录中的所有图像文件
    # Iterate through all image files in renders directory
    for fname in os.listdir(renders_dir):
        # 读取渲染图像和对应的真实图像
        # Read rendered image and corresponding ground truth image
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        
        # 转换为张量并移动到GPU，只保留RGB通道（前3个通道）
        # Convert to tensor and move to GPU, keep only RGB channels (first 3 channels)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    """
    评估模型性能，计算SSIM、PSNR和LPIPS指标
    Evaluate model performance, calculate SSIM, PSNR and LPIPS metrics
    
    Args:
        model_paths: 模型路径列表 / List of model paths
    """

    # 初始化结果字典
    # Initialize result dictionaries
    full_dict = {}  # 完整结果字典 / Full results dictionary
    per_view_dict = {}  # 每视图结果字典 / Per-view results dictionary
    full_dict_polytopeonly = {}  # 仅polytope的完整结果字典 / Full results dictionary for polytope only
    per_view_dict_polytopeonly = {}  # 仅polytope的每视图结果字典 / Per-view results dictionary for polytope only
    print("")

    # 遍历每个场景目录
    # Iterate through each scene directory
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            
            # 初始化当前场景的结果字典
            # Initialize result dictionaries for current scene
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # 获取测试目录路径
            # Get test directory path
            test_dir = Path(scene_dir) / "test"

            # 遍历测试目录中的每个方法
            # Iterate through each method in test directory
            for method in os.listdir(test_dir):
                print("Method:", method)

                # 初始化当前方法的结果字典
                # Initialize result dictionaries for current method
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                # 设置方法目录路径
                # Set method directory paths
                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"  # 真实图像目录 / Ground truth directory
                renders_dir = method_dir / "renders"  # 渲染图像目录 / Renders directory
                
                # 读取图像
                # Read images
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # 初始化指标列表
                # Initialize metric lists
                ssims = []  # SSIM指标列表 / SSIM metrics list
                psnrs = []  # PSNR指标列表 / PSNR metrics list
                lpipss = []  # LPIPS指标列表 / LPIPS metrics list

                # 计算每张图像的指标
                # Calculate metrics for each image
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))  # 计算SSIM / Calculate SSIM
                    psnrs.append(psnr(renders[idx], gts[idx]))  # 计算PSNR / Calculate PSNR
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))  # 计算LPIPS / Calculate LPIPS

                # 打印平均指标结果
                # Print average metric results
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                # 保存完整结果（平均值）
                # Save full results (averages)
                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                
                # 保存每视图结果（详细值）
                # Save per-view results (detailed values)
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            # 保存结果到JSON文件
            # Save results to JSON files
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # 设置GPU设备
    # Set up GPU device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 设置命令行参数解析器
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])  # 模型路径列表 / List of model paths
    
    # 解析命令行参数并执行评估
    # Parse command line arguments and execute evaluation
    args = parser.parse_args()
    evaluate(args.model_paths)
