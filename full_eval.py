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
import os
from argparse import ArgumentParser
import time

# 定义各种数据集的场景列表
# Define scene lists for various datasets
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]  # MipNeRF360户外场景 / MipNeRF360 outdoor scenes
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]  # MipNeRF360室内场景 / MipNeRF360 indoor scenes  
tanks_and_temples_scenes = ["truck", "train"]  # Tanks and Temples数据集场景 / Tanks and Temples dataset scenes
deep_blending_scenes = ["drjohnson", "playroom"]  # Deep Blending数据集场景 / Deep Blending dataset scenes

# 设置命令行参数解析器
# Set up command line argument parser
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")  # 跳过训练阶段 / Skip training phase
parser.add_argument("--skip_rendering", action="store_true")  # 跳过渲染阶段 / Skip rendering phase  
parser.add_argument("--skip_metrics", action="store_true")  # 跳过指标计算阶段 / Skip metrics calculation phase
parser.add_argument("--output_path", default="./eval")  # 输出路径 / Output path
parser.add_argument("--use_depth", action="store_true")  # 使用深度信息 / Use depth information
parser.add_argument("--use_expcomp", action="store_true")  # 使用曝光补偿 / Use exposure compensation
parser.add_argument("--fast", action="store_true")  # 快速模式（使用稀疏Adam） / Fast mode (use sparse Adam)
parser.add_argument("--aa", action="store_true")  # 抗锯齿 / Anti-aliasing




# 解析已知参数
# Parse known arguments
args, _ = parser.parse_known_args()

# 合并所有场景列表
# Combine all scene lists
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

# 如果需要训练或渲染，则添加数据集路径参数
# Add dataset path arguments if training or rendering is needed
if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)  # MipNeRF360数据集路径 / MipNeRF360 dataset path
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)  # Tanks and Temples数据集路径 / Tanks and Temples dataset path
    parser.add_argument("--deepblending", "-db", required=True, type=str)  # Deep Blending数据集路径 / Deep Blending dataset path
    args = parser.parse_args()

# 训练阶段
# Training phase
if not args.skip_training:
    # 设置通用训练参数
    # Set common training arguments
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 "
    
    # 根据选项添加额外参数
    # Add additional arguments based on options
    if args.aa:
        common_args += " --antialiasing "  # 添加抗锯齿参数 / Add anti-aliasing argument
    if args.use_depth:
        common_args += " -d depths2/ "  # 添加深度目录参数 / Add depth directory argument

    if args.use_expcomp:
        # 添加曝光补偿相关参数 / Add exposure compensation related arguments
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "

    if args.fast:
        # 使用稀疏Adam优化器进行快速训练 / Use sparse Adam optimizer for fast training
        common_args += " --optimizer_type sparse_adam "

    # 训练MipNeRF360户外场景
    # Train MipNeRF360 outdoor scenes
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        # 使用images_4作为图像输入目录（较低分辨率，适合户外场景）
        # Use images_4 as image input directory (lower resolution, suitable for outdoor scenes)
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    
    # 训练MipNeRF360室内场景  
    # Train MipNeRF360 indoor scenes
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        # 使用images_2作为图像输入目录（较高分辨率，适合室内场景）
        # Use images_2 as image input directory (higher resolution, suitable for indoor scenes)
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    m360_timing = (time.time() - start_time)/60.0  # 记录MipNeRF360训练时间（分钟） / Record MipNeRF360 training time in minutes

    # 训练Tanks and Temples场景
    # Train Tanks and Temples scenes
    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    tandt_timing = (time.time() - start_time)/60.0  # 记录Tanks and Temples训练时间（分钟） / Record Tanks and Temples training time in minutes

    # 训练Deep Blending场景
    # Train Deep Blending scenes
    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    db_timing = (time.time() - start_time)/60.0  # 记录Deep Blending训练时间（分钟） / Record Deep Blending training time in minutes

# 将训练时间写入文件
# Write training times to file
with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
    file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

# 渲染阶段
# Rendering phase
if not args.skip_rendering:
    # 构建所有数据集的源路径列表
    # Build source path list for all datasets
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    
    # 设置渲染的通用参数
    # Set common arguments for rendering
    common_args = " --quiet --eval --skip_train"
    
    # 根据选项添加额外参数
    # Add additional arguments based on options
    if args.aa:
        common_args += " --antialiasing "  # 添加抗锯齿参数 / Add anti-aliasing argument
    if args.use_expcomp:
        common_args += " --train_test_exp "  # 添加曝光补偿参数 / Add exposure compensation argument

    # 对每个场景进行渲染（7000和30000迭代）
    # Render each scene (7000 and 30000 iterations)
    for scene, source in zip(all_scenes, all_sources):
        # 渲染7000迭代的模型 / Render model at 7000 iterations
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        # 渲染30000迭代的模型 / Render model at 30000 iterations
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

# 指标计算阶段
# Metrics calculation phase
if not args.skip_metrics:
    # 构建所有场景路径的字符串
    # Build string of all scene paths
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    # 运行指标计算脚本，计算SSIM、PSNR和LPIPS
    # Run metrics calculation script to compute SSIM, PSNR and LPIPS
    os.system("python metrics.py -m " + scenes_string)
