export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

# GLUE datasets supported in this code (change in $TASK)
# MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB
# PEFT methods supported in this code (change in $MODE)
# loretta_adp, loretta_rep, lora, adapters (series), prompt, ia3, ptune
# change the number of cuda device by set the $DEVICE

# Test with deberta-base model (use SST2 dataset by default, change the input for $TASK for other tasks)

# MODEL=microsoft/deberta-base TASK=MNLI MODE=lora EPOCH=10 BS=32 LR=3e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=SST2 MODE=lora EPOCH=10 BS=32 LR=1e-4 DEVICE=0 bash finetune.sh

# MODEL=microsoft/deberta-base TASK=COLA MODE=lora EPOCH=30 BS=32 LR=4e-4 DEVICE=0 bash finetune.sh

# MODEL=microsoft/deberta-base TASK=MRPC MODE=lora EPOCH=10 BS=32 LR=4e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=QNLI MODE=lora EPOCH=25 BS=32 LR=3e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=QQP MODE=lora EPOCH=10 BS=16 LR=3e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=RTE MODE=lora EPOCH=50 BS=32 LR=4e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=STSB MODE=lora EPOCH=30 BS=16 LR=4e-4 DEVICE=0 bash finetune.sh


# Test with roberta-base model (use SST2 dataset by default, change the input for $TASK for other tasks)


# MODEL=roberta-base TASK=MNLI MODE=lora EPOCH=30 BS=16 LR=5e-4 DEVICE=0 bash finetune.sh



# MODEL=microsoft/deberta-v3-base TASK=COLA MODE=newlora EPOCH=25 BS=32 LR=5e-4 DEVICE=0 bash finetune.sh



# MODEL=microsoft/deberta-base TASK=MNLI MODE=lora EPOCH=10 BS=32 LR=3e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=SST2 MODE=newlora EPOCH=10 BS=32 LR=1e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=MRPC MODE=newlora EPOCH=10 BS=32 LR=4e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=COLA MODE=newlora EPOCH=30 BS=32 LR=4e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=QNLI MODE=newlora EPOCH=25 BS=32 LR=3e-4 DEVICE=0 bash finetune.sh
# # MODEL=microsoft/deberta-base TASK=QQP MODE=lora EPOCH=10 BS=16 LR=3e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=RTE MODE=newlora EPOCH=50 BS=32 LR=4e-4 DEVICE=0 bash finetune.sh
# MODEL=microsoft/deberta-base TASK=STSB MODE=lora EPOCH=30 BS=16 LR=4e-4 DEVICE=0 bash finetune.sh









MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh



# MODEL=microsoft/deberta-v3-base TASK=MNLI MODE=adalora EPOCH=7 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh

# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh
# MODEL=microsoft/deberta-v3-base TASK=QNLI MODE=adalora EPOCH=5 BS=32 LR=1.2e-3 DEVICE=0 RANK=8 bash finetune.sh


