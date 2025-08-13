export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=QQP MODE=pissa EPOCH=5 BS=16 LR=1e-4 DEVICE=0 RANK=8 bash finetune.sh