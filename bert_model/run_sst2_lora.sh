export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=SST2 MODE=lora EPOCH=20 BS=16 LR=3e-5 DEVICE=0 RANK=8 bash finetune.sh