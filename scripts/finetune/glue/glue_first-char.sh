#!/bin/bash

# specify job-related names
PRETRAIN_NAME=first-char
MODEL_NAME=roberta-base
NUM_STEPS=500k
MODEL_PATH=./weights/roberta-base_first-char_2e-4_42/checkpoint-500000

declare -a task_names=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb")
declare -a task_metrics=("matthews_correlation" "accuracy" "accuracy" "accuracy" "f1" "accuracy" "accuracy" "spearmanr")
declare -a task_logging_steps=("54" "2455" "23" "655" "2275" "16" "421" "36")

for seed in 42 22 29 97 96
  do
    for index in {0..7}
      do 
        # parameter settings
        task_name=${task_names[$index]}
        task_metric=${task_metrics[$index]}
        task_logging_step=${task_logging_steps[$index]}
        JOB_NAME=${MODEL_NAME}_${PRETRAIN_NAME}_${NUM_STEPS}_${task_name}_${seed}

        echo -e "\n\n**********"
        echo -e "Now running ${JOB_NAME}..."
        echo -e "**********\n\n"

        # run a GLUE script
        python src/run_glue.py \
          --model_name_or_path=${MODEL_PATH} \
          --tokenizer_name=${MODEL_NAME} \
          --task_name $task_name \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 64 \
          --seed=$seed \
          --learning_rate 3e-5 \
          --num_train_epochs 20 \
          --warmup_ratio=0.06 \
          --logging_steps=10 \
          --eval_steps=${task_logging_step} \
          --evaluation_strategy=steps \
          --save_steps=${task_logging_step} \
          --save_strategy=steps \
          --save_total_limit 2 \
          --output_dir ./weights/glue/${JOB_NAME} \
          --overwrite_output_dir \
          --logging_dir=./logs/glue/${JOB_NAME} \
          --disable_tqdm=True \
          --fp16 \
          --patience=5 \
          --metric_name=${task_metric} \
          --objective_type=maximize \
          --load_best_model_at_end \
          --metric_for_best_model=${task_metric} \
          --greater_is_better=True

        echo -e "\n***** ${JOB_NAME} Completed! *****\n\n\n"
      done
  done
