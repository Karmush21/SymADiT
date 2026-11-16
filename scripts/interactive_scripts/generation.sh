#!/usr/bin/env bash


ROOT_DIR=""
DATA_DIR="$ROOT_DIR/data/mp20"

LDM_CKPT="$ROOT_DIR/"

# Usually you don't need to specify a VAE checkpoint, as the LDM checkpoint already contains the VAE weights. If you want to use a different VAE, you can specify it here.
#VAE_CKPT=""


# ---------------------------------------------------------
# GPU setup
# ---------------------------------------------------------

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ No GPUs detected!"
    exit 1
fi

echo "Root dir: $ROOT_DIR"
echo "Data dir: $DATA_DIR"
echo "LDM checkpoint: $LDM_CKPT"
echo "Detected $NUM_GPUS GPU(s)"

export MASTER_PORT=${MASTER_PORT:-29521}


# ---------------------------------------------------------
# Generation
#
# Keep one process only. Generation/postprocessing writes
# files, so DDP is unnecessary here.
# ---------------------------------------------------------

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  src/generation.py \
  \
  ldm_ckpt="$LDM_CKPT" \
  vae_ckpt="$VAE_CKPT" \
  \
  paths.root_dir="$ROOT_DIR" \
  paths.data_dir="$DATA_DIR" \
  \
  logger.wandb.mode=online \
  logger.wandb.project=generation_interactive \
  task_name=generation_interactive \
  name=generation_interactive \
  \
  generate.num_samples=10 \
  generate.batch_size=5 \
  generate.cfg_scale=2 \
  generate.device=cuda \
  \
  conditioning.allowed_spacegroups=null \
  conditioning.forbidden_spacegroups=null \
  \
  seed=42 \
  \
  "$@"