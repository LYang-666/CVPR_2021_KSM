#!/bin/bash
# Usage:
# source ./scripts/run.sh resnet50 1 1 

# This is hard-coded to prevent silly mistakes.
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imagenet"]="1000"
NUM_OUTPUTS["places"]="365"
NUM_OUTPUTS["stanford_cars_cropped"]="196"
NUM_OUTPUTS["cubs_cropped"]="200"
NUM_OUTPUTS["flowers"]="102"
NUM_OUTPUTS["wikiart"]="195"
NUM_OUTPUTS["sketches"]="250"

ARCH=$1
GPU_ID=$2
NUM_RUNS=$3
MASK_SCALE=1e-2
LR_MASK=2e-4
LR_CLASS=2e-4
NUM_EPOCHS=30
# {"stanford_cars_cropped" "cubs_cropped" "flowers" "wikiart" "sketches"}
for RUN_ID in `seq 1 $NUM_RUNS`;
do
  # for DATASET in wikiart sketches flowers cubs_cropped stanford_cars_cropped; do
  for DATASET in stanford_cars_cropped; do
    mkdir ../checkpoints/test
    mkdir ../logs/test

    TAG=$ARCH'_'$MASK_SCALE'_lr'$LR_MASK'-'$LR_CLASS'_decay'$MASK_DECAY'-'$CLASS_DECAY'_'$RUN_ID

    CKPT_DIR='../checkpoints/'$DATASET'/test/'
    mkdir -p $CKPT_DIR
    LOG_DIR='../logs/'$DATASET'/test/'
    mkdir -p $LOG_DIR

    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --mode finetune \
      --arch $ARCH  \
      --train_bn \
      --mask_scale $MASK_SCALE \
      --dataset $DATASET --num_outputs ${NUM_OUTPUTS[$DATASET]} '--mask_adam' \
      --lr_mask $LR_MASK  \
      --lr_classifier $LR_CLASS  \
      --finetune_epochs $NUM_EPOCHS \
      --save_prefix $CKPT_DIR$TAG'.pt' | tee $LOG_DIR$TAG'.txt'
  done
done
