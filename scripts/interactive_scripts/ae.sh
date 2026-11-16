#!/usr/bin/env bash


# ---------------------------------------------------------
# Auto-detect environment
# ---------------------------------------------------------



ROOT_DIR="/"
DATA_DIR="/"





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
  src/train_autoencoder.py \
  \
  paths.root_dir="$ROOT_DIR" \
  paths.data_dir="$DATA_DIR" \
  \
  name=ae_interactive \
  task_name=ae_interactive \
  logger.wandb.project=ae_interactive \
  logger.wandb.mode=online \
  \
  trainer.check_val_every_n_epoch=100 \
  trainer.log_every_n_steps=1 \
  trainer.max_epochs=500 \
  \
  data.datamodule.batch_size.train=1 \
  data.datamodule.batch_size.val=1 \
  data.datamodule.batch_size.test=1 \
  \
  data.datamodule.num_workers.train=0 \
  data.datamodule.num_workers.val=0 \
  data.datamodule.num_workers.test=0 \
  \
  autoencoder_module.latent_dim=32 \
  autoencoder_module.use_regularized_kl=true \
  autoencoder_module.use_regularized_ae=false \
  \
  autoencoder_module.loss_weights.loss_kl=0.0001 \
  autoencoder_module.loss_weights.loss_pos=5.0 \
  \
  autoencoder_module.augmentations.do_augmentations=false \
  \
  \
  autoencoder_module.pos_loss=periodic \
  autoencoder_module.cell_loss=l2 \
  \
  debug=true \
  \
  "$@"