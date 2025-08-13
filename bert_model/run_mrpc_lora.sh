export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=MRPC MODE=lora EPOCH=20 BS=32 LR=2e-4 DEVICE=0 RANK=8 bash finetune.sh