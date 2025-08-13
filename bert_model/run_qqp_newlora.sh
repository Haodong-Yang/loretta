export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=QQP MODE=newlora EPOCH=10 BS=16 LR=1e-4 DEVICE=0 RANK=8 TARGET=64 KEEP=0.5 bash finetune.sh