export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base TASK=MNLI MODE=lora EPOCH=7 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
