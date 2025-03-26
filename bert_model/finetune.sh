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
PATH=${PATH:-sst2}
TASK=${TASK:-sst2}
export CUDA_VISIBLE_DEVICES=$DEVICE
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_P2P_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

current_path=$(pwd)
NUM_GPUS=$(echo $DEVICE | awk -F',' '{print NF}')
python -m torch.distributed.run --nproc_per_node=$NUM_GPUS run_glue_v5.py \
  --data_dir=default \
  --logging_dir="$current_path/logs/$TASK-$BS-$LR-$MODEL_NAME-$MODE-${20}-$(date +"%Y%m%d%H%M%S")" \
  --model_name_or_path=$MODEL --tokenizer_name=$MODEL --evaluation_strategy=steps --eval_steps=500 --logging_steps=50 \
  --overwrite_output_dir --save_steps=10000 --task_name=$TASK --warmup_step=500 --learning_rate=$LR \
  --num_train_epochs=$EPOCH --per_device_train_batch_size=$BS --output_dir="$current_pathd/$PATH" --max_seq_length=128 \
  --tuning_type=$MODE --do_train --do_eval --tensor_rank=$RANK


