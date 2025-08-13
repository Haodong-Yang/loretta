export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=MNLI MODE=newlora EPOCH=5 BS=16 LR=5e-4 DEVICE=0 RANK=8 TARGET=256 KEEP=0.5 bash finetune.sh