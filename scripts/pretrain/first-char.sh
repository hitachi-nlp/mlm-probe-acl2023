#!/bin/bash

TASK_NAME=first-char
MODEL_NAME=roberta-base
SEED=42
LEARNING_RATE=2e-4
JOB_NAME=${MODEL_NAME}_${TASK_NAME}_${LEARNING_RATE}_${SEED}

python -m torch.distributed.launch \
    --master_port 1235 \
    --nproc_per_node 4 \
    --nnodes 1 \
src/pretrainer.py \
    --model_name_or_path=${MODEL_NAME} \
    --wiki_data_dir=/path/to/wikipedia/data \
    --book_data_dir=/path/to/bookcorpus \
    --save_interval=43200.0 \
    --learning_rate=${LEARNING_RATE} \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --max_grad_norm=1.0 \
    --max_steps=500000 \
    --warmup_steps=10000 \
    --save_steps=100000 \
    --seed=${SEED} \
    --per_device_train_batch_size=32 \
    --logging_steps=50 \
    --output_dir=./weights/${JOB_NAME} \
    --overwrite_output_dir \
    --logging_dir=./logs/${JOB_NAME} \
    --disable_tqdm=True \
    --fp16
