#!/usr/bin/env bash
# =============================================================================
# TCIR 北太平洋(WPAC) 强度回归 —— 一键训练脚本
# -----------------------------------------------------------------------------
# 用法:
#   ./train.sh                       # 默认: vgg16 骨干, 40 轮
#   ./train.sh resnet18             # 换 resnet18 骨干
#   ./train.sh vgg16 60             # vgg16, 训练 60 轮
#   ./train.sh vgg16 40 train.lr=2e-4 train.batch_size=32   # 追加任意 --set 覆盖
#
# 说明:
#   - 自动使用 dl_env 虚拟环境的 python (无需手动 conda activate)
#   - 实验名 / 输出目录按骨干自动区分, 互不覆盖
#   - 数据已抽取为 data/wpac_96.h5 + data/wpac_compact.csv, 无需碰 30GB 原文件
# =============================================================================
set -euo pipefail

# ----------------------------- 环境路径 ---------------------------------------
PROJECT_DIR="/home/yzm/tf/tc_intensity"
PYTHON="/home/yzm/tf/dl-env/bin/python3.11"
CONFIG="configs/tcir_wpac_train.yaml"

# ----------------------------- 可调参数 ---------------------------------------
BACKBONE="${1:-vgg16}"            # 默认 vgg16 (与当前改造一致)
EPOCHS="${2:-40}"                 # 默认 40 轮
EXTRA_OVERRIDES=("${@:3}")       # 其余参数原样传给 --set

# 按骨干自动命名, 避免不同实验互相覆盖
EXP_NAME="tcir_wpac_${BACKBONE}"
OUTPUT_DIR="outputs/tcir_wpac_${BACKBONE}"

# ----------------------------- 前置检查 ---------------------------------------
if [ ! -x "$PYTHON" ]; then
    echo "❌ 找不到 python: $PYTHON" >&2
    exit 1
fi
if [ ! -f "$PROJECT_DIR/$CONFIG" ]; then
    echo "❌ 找不到配置: $PROJECT_DIR/$CONFIG" >&2
    exit 1
fi
if [ ! -f "$PROJECT_DIR/data/wpac_96.h5" ]; then
    echo "❌ 找不到数据: $PROJECT_DIR/data/wpac_96.h5 (先运行 scripts/extract_wpac.py)" >&2
    exit 1
fi

# ----------------------------- 执行训练 ---------------------------------------
cd "$PROJECT_DIR"
echo "==> 项目目录 : $PROJECT_DIR"
echo "==> Python   : $PYTHON"
echo "==> 骨干     : $BACKBONE"
echo "==> 轮数     : $EPOCHS"
echo "==> 实验名   : $EXP_NAME"
echo "==> 输出目录 : $OUTPUT_DIR"
echo "==> 额外覆盖 : ${EXTRA_OVERRIDES[*]:-无}"
echo "------------------------------------------------------------------------"

"$PYTHON" train.py -c "$CONFIG" \
    --set "model.backbone=$BACKBONE" \
           "train.epochs=$EPOCHS" \
           "experiment.name=$EXP_NAME" \
           "experiment.output_dir=$OUTPUT_DIR" \
           "${EXTRA_OVERRIDES[@]}"

echo "------------------------------------------------------------------------"
echo "✅ 训练完成. 最佳权重: $OUTPUT_DIR/$EXP_NAME/checkpoints/best.ckpt"
