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
import logging
from argparse import ArgumentParser
import shutil

# 这个Python脚本基于MipNerF 360存储库中提供的shell转换器脚本
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')  # 禁用GPU / Disable GPU
parser.add_argument("--skip_matching", action='store_true')  # 跳过特征匹配步骤 / Skip feature matching step
parser.add_argument("--source_path", "-s", required=True, type=str)  # 源路径（必需） / Source path (required)
parser.add_argument("--camera", default="OPENCV", type=str)  # 相机模型类型 / Camera model type
parser.add_argument("--colmap_executable", default="", type=str)  # COLMAP可执行文件路径 / COLMAP executable path
parser.add_argument("--resize", action="store_true")  # 启用图像调整大小 / Enable image resizing
parser.add_argument("--magick_executable", default="", type=str)  # ImageMagick可执行文件路径 / ImageMagick executable path

# 解析命令行参数
# Parse command line arguments
args = parser.parse_args()

# 设置COLMAP和ImageMagick命令
# Set up COLMAP and ImageMagick commands
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0  # GPU使用标志 / GPU usage flag

# 如果不跳过特征匹配，则执行COLMAP的SfM流水线
# If not skipping feature matching, execute COLMAP's SfM pipeline
if not args.skip_matching:
    # 创建distorted/sparse目录用于存储稀疏重建结果
    # Create distorted/sparse directory for storing sparse reconstruction results
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    # 特征提取
    # Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 特征匹配
    # Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 束光法平差（Bundle Adjustment）
    # Bundle adjustment
    # 默认的Mapper容差过大，减小它可以加速束光法平差步骤
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# 图像去畸变
# Image undistortion
# 我们需要将图像去畸变为理想的针孔相机内参
# We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# 整理稀疏重建文件
# Organize sparse reconstruction files
files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# 将每个文件从源目录复制到目标目录
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# 如果启用了调整大小选项，则创建多分辨率图像
# If resize option is enabled, create multi-resolution images
if(args.resize):
    print("Copying and resizing...")

    # 调整图像大小，创建多个分辨率版本
    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)  # 50%分辨率 / 50% resolution
    os.makedirs(args.source_path + "/images_4", exist_ok=True)  # 25%分辨率 / 25% resolution
    os.makedirs(args.source_path + "/images_8", exist_ok=True)  # 12.5%分辨率 / 12.5% resolution
    
    # 获取源目录中的文件列表
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    
    # 将每个文件从源目录复制到目标目录并调整大小
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        # 创建50%分辨率版本
        # Create 50% resolution version
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 创建25%分辨率版本
        # Create 25% resolution version
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 创建12.5%分辨率版本
        # Create 12.5% resolution version
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

# 处理完成
# Processing complete
print("Done.")
