export HF_HOME="/scratch/pawsey1001/haodongyang/experiment/cache"
export WANDB_START_METHOD="thread"

MODEL=microsoft/deberta-v3-base 
TASK=RTE 
MODE=lora 
EPOCH=50 
BS=32 
LR=1.2e-4
DEVICE=0 
RANK=8
TRANK=1
SEED=6
COEF=0.3
init_warmup=600
final_warmup=1800 
mask_interval=1
lora_alpha=32
max_seq_length=320
# --beta1 0.85 --beta2 0.85


# --cls_dropout 0.20 --weight_decay 0.01



MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

EPOCH=${EPOCH:-5}
BS=${BS:-4}
LR=${LR:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
RANK=${RANK:-8}
MODE=${MODE:-ft}
DEVICE=${DEVICE:-0}
OUTPUT_PATH=${TASK:-sst2}
TASK=${TASK:-sst2}
export HIP_VISIBLE_DEVICES=$DEVICE

current_path=$(pwd)
python run_glue_v5.py \
  --data_dir=default \
  --logging_dir="$current_path/logs/$TASK-$BS-$LR-$MODEL_NAME-$MODE-${20}-$(date +"%Y%m%d%H%M%S")" \
  --model_name_or_path=$MODEL --tokenizer_name=$MODEL --evaluation_strategy=steps --eval_steps=100 --logging_steps=10 \
  --overwrite_output_dir --save_strategy=steps --save_steps=10000 --task_name=$TASK --warmup_step=200 --learning_rate=$LR \
  --num_train_epochs=$EPOCH --per_device_train_batch_size=$BS --output_dir="$current_path/output/$MODEL_NAME/$MODE/$OUTPUT_PATH-$(date +"%Y%m%d%H%M%S")" --max_seq_length=128 \
  --tuning_type=$MODE --do_train --do_eval --do_predict --target_rank=$TRANK --lora_r=$RANK \
  --coef=$COEF --init_warmup=$init_warmup --final_warmup=$final_warmup --mask_interval=$mask_interval \
  --seed=$SEED --lora_alpha=$lora_alpha


