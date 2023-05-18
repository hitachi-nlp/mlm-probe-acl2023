#!/bin/bash

# specify job-related params
PRETRAIN_NAME=last-char
MODEL_NAME=roberta-base
NUM_STEPS=500k
MODEL_PATH=./weights/roberta-base_last-char_1e-4_42/checkpoint-500000


for seed in 42 22 29 97 96
  do
    JOB_NAME=${MODEL_NAME}_${PRETRAIN_NAME}_${NUM_STEPS}_squad_${seed}
    echo -e "\n\n**********"
    echo -e "Now running ${JOB_NAME}..."
    echo -e "**********\n\n"

    # run a SQuAD script
    python src/run_squad.py \
      --model_name_or_path=${MODEL_PATH} \
      --tokenizer_name=${MODEL_NAME} \
      --dataset_name squad \
      --do_train \
      --do_eval \
      --seed $seed \
      --per_device_train_batch_size 24 \
      --per_device_eval_batch_size 48 \
      --learning_rate 3e-5 \
      --num_train_epochs 10 \
      --max_seq_length 384 \
      --doc_stride 128 \
      --disable_tqdm=True \
      --warmup_ratio=0.06 \
      --logging_steps=10 \
      --eval_steps=185 \
      --save_steps=185 \
      --evaluation_strategy=steps \
      --save_strategy=steps \
      --save_total_limit 2 \
      --output_dir ./weights/squad/${JOB_NAME} \
      --overwrite_output_dir \
      --logging_dir=./logs/squad/${JOB_NAME} \
      --fp16 \
      --patience=5 \
      --metric_name=f1 \
      --objective_type=maximize \
      --load_best_model_at_end \
      --metric_for_best_model=f1 \
      --greater_is_better=True

      echo -e "\n***** ${JOB_NAME} Completed! *****\n\n\n"
  done