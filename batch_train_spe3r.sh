#!/bin/bash

# 批量训练 SPE3R 数据集中所有航天器的 Bash 脚本
# 使用方法：./batch_train_spe3r.sh [start_from_spacecraft]

DATASET_BASE_PATH="/teams/microsate_1687685838/qza/dataset/SPE3R/spe3r-colmap"
OUTPUT_BASE_PATH="/teams/microsate_1687685838/qza/output"

# 所有航天器列表
SPACECRAFT=(
    "acrimsat_final"
    "apollo_soyuz_carbajal"
    "aqua"
    "aquarius_dep"
    "aquarius_ud"
    "aura_v002"
    "bepi_mpo"
    "bepi_mtm"
    "calipso_v016_trip"
    "cassini_66"
    "chandra_v09"
    "cheops"
    "clementine_v01"
    "cloudsat_v19"
    "cygnss_solo_39"
    "dawn_19"
    "deep_space_1_11"
    "double_star"
    "eo-1_final"
    "galileo_actual_05"
    "giotto"
    "grace_v011"
    "herschel"
    "hst"
    "icesat_v015_uv_final"
    "integral"
    "jason-1_final"
    "juice"
    "juno"
    "kepler_v009"
    "loral-1300com-main"
    "lro_35"
    "magellan_16"
    "mars_global_surveyor"
    "mars_odyssey"
    "maven"
    "messenger"
    "mro_13"
    "near_07"
    "npp_16"
    "ostm_jason-2"
    "parkersolarprobe"
    "philae"
    "pioneer"
    "planck"
    "proba_2"
    "proba_3_csc"
    "proba_3_osc"
    "quikscat_v006"
    "rosetta"
    "smart_1"
    "soho"
    "solar_orbiter"
    "sorce_vfinal"
    "stardust_35"
    "tdrs"
    "terra_vfinal"
    "tess"
    "tgo_edm"
    "themis"
    "trmm_vfinal"
    "venus_express"
    "voyager_17"
    "xmm_newton"
)

# 记录函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查是否从指定航天器开始
START_FROM="$1"
START_TRAINING=false

if [ -z "$START_FROM" ]; then
    START_TRAINING=true
    log_message "Starting batch training for all ${#SPACECRAFT[@]} spacecraft..."
else
    log_message "Starting batch training from: $START_FROM"
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE_PATH"

# 统计变量
SUCCESSFUL=0
FAILED=0
SKIPPED=0
TOTAL_START_TIME=$(date +%s)

# 遍历所有航天器
for i in "${!SPACECRAFT[@]}"; do
    SPACECRAFT_NAME="${SPACECRAFT[$i]}"
    
    # 检查是否到达起始点
    if [ "$START_TRAINING" = false ]; then
        if [ "$SPACECRAFT_NAME" = "$START_FROM" ]; then
            START_TRAINING=true
            log_message "Reached starting point: $START_FROM"
        else
            continue
        fi
    fi
    
    CURRENT_NUM=$((i + 1))
    log_message ""
    log_message "=================================================="
    log_message "Training ${CURRENT_NUM}/${#SPACECRAFT[@]}: $SPACECRAFT_NAME"
    log_message "=================================================="
    
    DATASET_PATH="$DATASET_BASE_PATH/$SPACECRAFT_NAME"
    OUTPUT_PATH="$OUTPUT_BASE_PATH/$SPACECRAFT_NAME"
    
    # 检查数据集是否存在
    if [ ! -d "$DATASET_PATH" ]; then
        log_message "❌ Dataset not found: $DATASET_PATH"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # 检查输出目录是否已存在
    if [ -d "$OUTPUT_PATH" ]; then
        log_message "⏭️  Output directory already exists for $SPACECRAFT_NAME, skipping..."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_PATH"
    
    # 开始训练
    log_message "Starting training for $SPACECRAFT_NAME..."
    START_TIME=$(date +%s)
    
    # 执行训练命令
    python train.py -s "$DATASET_PATH" -m "$OUTPUT_PATH"
    RESULT=$?
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $RESULT -eq 0 ]; then
        log_message "✅ Successfully completed training for $SPACECRAFT_NAME in ${DURATION}s"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        log_message "❌ Training failed for $SPACECRAFT_NAME"
        FAILED=$((FAILED + 1))
    fi
done

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

# 打印最终统计
log_message ""
log_message "============================================================"
log_message "BATCH TRAINING COMPLETED"
log_message "============================================================"
log_message "Total time: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m)"
log_message "Successfully trained: $SUCCESSFUL"
log_message "Failed: $FAILED"
log_message "Skipped: $SKIPPED"
log_message "Total processed: $((SUCCESSFUL + FAILED + SKIPPED))"
