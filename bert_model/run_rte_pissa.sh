export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=RTE MODE=pissa EPOCH=50 BS=16 LR=5e-5 DEVICE=0 RANK=8 bash finetune.sh