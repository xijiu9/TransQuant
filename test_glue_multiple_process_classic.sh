# Make sure run "$accelerate config" before you run this script

# Example setting:
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 0
# Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]: NO
# Do you want to use DeepSpeed? [yes/NO]: NO
# Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: NO

# export TASK_NAME=qnli
# 小任务使用单卡训练，在少量epoch下获得更好的表现
export CUDA_VISIBLE_DEVICES=3

# for SEED in 21 42 87
for SEED in 21 42 87
do
accelerate launch test_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name $3 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --seed $SEED \
    --num_train_epochs 3 \
    --output_dir ./test_glue_result/test/$3/seed=${SEED} \
    --arch $1 \
    $2 
done