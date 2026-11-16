#!/usr/bin/env bash



ROOT_DIR=""
DATA_DIR=""
AUTOENCODER_CKPT="$ROOT_DIR/logs/"


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
echo "Detected $NUM_GPUS GPU(s)"

export MASTER_PORT=${MASTER_PORT:-29521}
 

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  src/train_diffusion.py \
  \
  diffusion_module.autoencoder_ckpt="$AUTOENCODER_CKPT" \
  diffusion_module.denoiser.d_x=32 \
  diffusion_module.x_loss=smooth_l1 \
  \
  paths.root_dir="$ROOT_DIR" \
  paths.data_dir="$DATA_DIR" \
  diffusion_module.sampling.data_dir="$DATA_DIR" \
  \
  name=ldm_interactive \
  task_name=ldm_interactive \
  logger.wandb.project=ldm_interactive \
  logger.wandb.mode=online \
  \
  trainer.check_val_every_n_epoch=5 \
  trainer.log_every_n_steps=1 \
  trainer.max_epochs=10 \
  \
  data.datamodule.batch_size.train=1 \
  data.datamodule.batch_size.val=1 \
  data.datamodule.batch_size.test=1 \
  \
  diffusion_module.sampling.batch_size=2 \
  diffusion_module.sampling.num_samples=4 \
  diffusion_module.conditioning.allowed_spacegroups=null \
  \
  data.datamodule.num_workers.train=0 \
  data.datamodule.num_workers.val=0 \
  data.datamodule.num_workers.test=0 \
  \
  diffusion_module.novelty.compute_novelty=false \
  \
  debug=true \
  test=false \
  \
  "$@"