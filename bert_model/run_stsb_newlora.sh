export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=STSB MODE=newlora EPOCH=20 BS=8 LR=3e-5 DEVICE=0 RANK=8 TARGET=32 KEEP=0.5 bash finetune.sh
