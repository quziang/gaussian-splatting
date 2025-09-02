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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# 尝试导入可选依赖库
# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    主训练函数 - 执行高斯散点的训练过程
    Main training function - Executes the training process for Gaussian splatting
    
    Args:
        dataset: 数据集参数 / Dataset parameters
        opt: 优化参数 / Optimization parameters  
        pipe: 渲染管道参数 / Rendering pipeline parameters
        testing_iterations: 测试迭代列表 / List of testing iterations
        saving_iterations: 保存迭代列表 / List of saving iterations
        checkpoint_iterations: 检查点迭代列表 / List of checkpoint iterations
        checkpoint: 检查点路径 / Checkpoint path
        debug_from: 调试起始迭代 / Debug starting iteration
    """

    # 检查稀疏Adam优化器的可用性
    # Check availability of sparse Adam optimizer
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 初始化训练设置
    # Initialize training setup
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)  # 准备输出目录和日志记录器 / Prepare output directory and logger
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)  # 创建高斯模型 / Create Gaussian model
    scene = Scene(dataset, gaussians)  # 创建场景 / Create scene
    gaussians.training_setup(opt)  # 设置训练参数 / Setup training parameters
    
    # 如果有检查点，则加载检查点
    # Load checkpoint if available
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色
    # Set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    # Create CUDA events for timing
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 设置优化器类型和深度权重函数
    # Setup optimizer type and depth weight function
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 初始化视点堆栈和索引
    # Initialize viewpoint stack and indices
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0  # 指数移动平均损失用于日志记录 / Exponential moving average loss for logging
    ema_Ll1depth_for_log = 0.0  # 指数移动平均深度损失用于日志记录 / Exponential moving average depth loss for logging

    # 创建进度条
    # Create progress bar
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 主训练循环
    # Main training loop
    for iteration in range(first_iter, opt.iterations + 1):
        # 网络GUI连接处理
        # Network GUI connection handling
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 开始计时
        # Start timing
        iter_start.record()

        # 更新学习率
        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数的阶数，直到最大阶数
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视点
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # 渲染
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 设置背景（随机背景或固定背景）
        # Set background (random or fixed)
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 执行渲染
        # Execute rendering
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 应用Alpha遮罩（如果存在）
        # Apply alpha mask if available
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # 计算损失
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()  # 获取真实图像 / Get ground truth image
        Ll1 = l1_loss(image, gt_image)  # 计算L1损失 / Calculate L1 loss
        
        # 计算SSIM损失
        # Calculate SSIM loss
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 总损失：L1损失和SSIM损失的加权组合
        # Total loss: weighted combination of L1 loss and SSIM loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 深度正则化
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]  # 渲染的逆深度图 / Rendered inverse depth map
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()  # 单目逆深度图 / Monocular inverse depth map
            depth_mask = viewpoint_cam.depth_mask.cuda()  # 深度遮罩 / Depth mask

            # 计算深度L1损失
            # Calculate depth L1 loss
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # 反向传播
        # Backward propagation
        loss.backward()

        # 结束计时
        # End timing
        iter_end.record()

        with torch.no_grad():
            # 更新进度条显示
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  # 指数移动平均 / Exponential moving average
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录日志和保存模型
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密集化处理
            # Densification
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中的最大半径用于修剪
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 密集化和修剪
                # Densification and pruning
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # 重置不透明度
                # Reset opacity
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            # Optimizer step
            if iteration < opt.iterations:
                # 曝光优化器步骤
                # Exposure optimizer step
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                
                # 根据优化器类型执行不同的优化步骤
                # Execute different optimization steps based on optimizer type
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            # Save checkpoint
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """
    准备输出目录和日志记录器
    Prepare output directory and logger
    
    Args:
        args: 命令行参数 / Command line arguments
    
    Returns:
        tb_writer: TensorBoard写入器 / TensorBoard writer
    """    
    # 如果没有指定模型路径，则自动生成一个唯一路径
    # If no model path is specified, automatically generate a unique path
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 创建输出文件夹
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # 保存配置参数到文件
    # Save configuration parameters to file
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard记录器
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    """
    训练报告函数 - 记录训练过程中的指标和测试结果
    Training report function - Log metrics and test results during training
    
    Args:
        tb_writer: TensorBoard写入器 / TensorBoard writer
        iteration: 当前迭代次数 / Current iteration
        Ll1: L1损失 / L1 loss
        loss: 总损失 / Total loss
        l1_loss: L1损失函数 / L1 loss function
        elapsed: 经过时间 / Elapsed time
        testing_iterations: 测试迭代列表 / Testing iterations list
        scene: 场景对象 / Scene object
        renderFunc: 渲染函数 / Render function
        renderArgs: 渲染参数 / Render arguments
        train_test_exp: 训练测试曝光 / Train test exposure
    """
    # 记录训练损失到TensorBoard
    # Log training loss to TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 在测试迭代时进行评估
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()  # 清空GPU缓存 / Clear GPU cache
        
        # 配置测试和训练样本
        # Configure test and training samples
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 对每个配置进行评估
        # Evaluate each configuration
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                # 遍历每个相机视点进行渲染和评估
                # Iterate through each camera viewpoint for rendering and evaluation
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染图像并限制像素值范围
                    # Render image and clamp pixel values
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # 如果使用训练测试曝光，则裁剪图像
                    # Crop image if using train test exposure
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    
                    # 记录前5张图像到TensorBoard
                    # Log first 5 images to TensorBoard
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    # 累积L1损失和PSNR
                    # Accumulate L1 loss and PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                # 计算平均值并打印结果
                # Calculate averages and print results
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # 记录评估指标到TensorBoard
                # Log evaluation metrics to TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 记录场景统计信息
        # Log scene statistics
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()  # 清空GPU缓存 / Clear GPU cache

if __name__ == "__main__":
    # 设置命令行参数解析器
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)  # 模型参数 / Model parameters
    op = OptimizationParams(parser)  # 优化参数 / Optimization parameters
    pp = PipelineParams(parser)  # 管道参数 / Pipeline parameters
    
    # 添加额外的命令行参数
    # Add additional command line arguments
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # GUI服务器IP地址 / GUI server IP address
    parser.add_argument('--port', type=int, default=6009)  # GUI服务器端口 / GUI server port
    parser.add_argument('--debug_from', type=int, default=-1)  # 调试起始迭代 / Debug starting iteration
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 检测异常 / Detect anomaly
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 测试迭代列表 / Test iterations list
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 保存迭代列表 / Save iterations list
    parser.add_argument("--quiet", action="store_true")  # 静默模式 / Quiet mode
    parser.add_argument('--disable_viewer', action='store_true', default=False)  # 禁用查看器 / Disable viewer
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 检查点迭代列表 / Checkpoint iterations list
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 起始检查点 / Starting checkpoint
    
    # 解析命令行参数
    # Parse command line arguments
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  # 添加最终迭代到保存列表 / Add final iteration to save list
    
    print("Optimizing " + args.model_path)

    # 初始化系统状态（随机数生成器）
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 启动GUI服务器，配置并运行训练
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 设置异常检测 / Set anomaly detection
    
    # 开始训练
    # Start training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 训练完成
    # All done
    print("\nTraining complete.")
