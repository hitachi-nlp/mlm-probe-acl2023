How does the task complexity of masked pretraining objectives affect downstream performance?
===

This is the official implementation of the paper titled "How does the task complexity of masked pretraining objectives affect downstream performance?". For reproduction, please follow the following procedures and have a look at [5. Reproduction](#5-reproduction).

## 1. Installation

### Requirements
* Python 3.9.2
* PyTorch 1.8.1
* CUDA 10.1

### PyTorch 
`pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

### Other packages
```bash
pip install -r requirements.txt
```

## 2. Preprocessing
**BookCorpus**
* Preprocessing from scratch  
  ```bash
  cd src/utils/preprocess
  python preprocessing_book.py /path/to/output/directory
  ```
  > It will take a while.

**Wikipedia**
* Preprocessing from scratch  
  ```bash
  cd src/utils/preprocess
  python preprocessing_wiki.py /path/to/output/directory
  ```
  > It will take a while.


## 3. Pretraining
### Usage
```
python src/pretrainer.py -h
```

### Masked First Character Prediction (First Char)

The following is an example of pretraining a BERT with First Char using four GPUs, which is the same setting as ours.

```bash
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
    --task_type first_char \
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
```

### Masked First-to-N Characters Prediction (First $n$ Chars)

The following is an example of pretraining a BERT with First 1 Char using four GPUs, which is the same setting as ours.

```bash
TASK_NAME=1-chars
MODEL_NAME=roberta-base
SEED=42
LEARNING_RATE=2e-4
JOB_NAME=${MODEL_NAME}_${TASK_NAME}_${LEARNING_RATE}_${SEED}

python -m torch.distributed.launch \
    --master_port 1239 \
    --nproc_per_node 4 \
    --nnodes 1 \
src/pretrainer.py \
    --model_name_or_path=${MODEL_NAME} \
    --task_type=n_chars \
    --wiki_data_dir=/path/to/wikipedia/data \
    --book_data_dir=/path/to/bookcorpus \
    --save_interval=43200.0 \
    --num_chars=1 \
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
```

### Masked Last-to-N Characters Prediction (Last $n$ Chars)

The following is an example of pretraining a BERT with Last 2 Chars using four GPUs, which is the same setting as ours.

```bash
TASK_NAME=last-2-chars
MODEL_NAME=roberta-base
SEED=42
LEARNING_RATE=2e-4
JOB_NAME=${MODEL_NAME}_${TASK_NAME}_${LEARNING_RATE}_${SEED}

python -m torch.distributed.launch \
    --master_port 1239 \
    --nproc_per_node 4 \
    --nnodes 1 \
src/pretrainer.py \
    --model_name_or_path=${MODEL_NAME} \
    --task_type=tail_n_chars \
    --wiki_data_dir=/path/to/wikipedia/data \
    --book_data_dir=/path/to/bookcorpus \
    --save_interval=43200.0 \
    --num_chars=1 \
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
```


## 4. Fine-tuning
### GLUE

The following is an example of fine-tuning a BERT (pre-trained with First 2 Chars) on GLUE, which is the same setting as ours.

```bash
# parameter settings
PRETRAIN_NAME=2-chars
MODEL_NAME=roberta-base
NUM_STEPS=500k
MODEL_PATH=./weights/roberta-base_${PRETRAIN_NAME}_2e-4_42/checkpoint-500000

task_name=cola
task_metric=matthews_correlation
task_logging_step=54
seed=42
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
  --logging_dir ./logs/glue/${JOB_NAME} \
  --disable_tqdm=True \
  --fp16 \
  --patience=5 \
  --metric_name=${task_metric} \
  --objective_type=maximize \
  --load_best_model_at_end \
  --metric_for_best_model=${task_metric} \
  --greater_is_better=True
```

### SQuAD

The following is an example of fine-tuning a BERT (pre-trained with First 9 Chars) on SQuAD, which is the same setting as ours.

```bash
PRETRAIN_NAME=9-chars
MODEL_NAME=roberta-base
NUM_STEPS=500k
MODEL_PATH=./weights/roberta-base_${PRETRAIN_NAME}_2e-4_42/checkpoint-500000
seed=42
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
```

### Dependency Parsing
We use the [SuPar library](https://github.com/yzhangcs/parser) to fine-tune models on Universal Dependencies (UD) datasets.

#### Download datasets
As explained in the paper, we use the English subset of Universal Dependencies (UD) v2.10.

Please download the dataset from [https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758) and unfreeze it.

#### Install the Supar library
```bash
pip install -U supar
```

#### Fine-tuning
The following is an example of fine-tuning a BERT (pre-trained with MLM) on UD Enslish EWT, which is the same setting as ours.

```bash
num_steps=500000
seed=42

# mlm
python -u -m supar.cmds.biaffine_dep train -b -d 0 -c ./scripts/finetune/ud/config/mlm.ini -p mlm_${num_steps}_${seed} \
	--seed=${seed} \
	--train /path/to/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-train.conllu \
	--dev /path/to/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-dev.conllu \
	--test /path/to/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-test.conllu \
	--encoder bert \
	--bert ./weights/roberta-base_mlm_2e-4_42/checkpoint-${num_steps}
```

## 5. Reproduction
### Training Scripts
We provide pre-training and fine-tuning bash scripts under [./scripts](./scripts). To use them, you must at least modify dataset and model paths, accordingly.

* Pretraining
  * [First Char](./scripts/pretrain/first-char.sh)
  * [Last Char](./scripts/pretrain/last-char.sh)
  * [First $n$ Chars](./scripts/pretrain/first-n-chars.sh)
  * [Last $n$ Chars](./scripts/pretrain/last-n-chars.sh)
  * [MLM](./scripts/pretrain/mlm.sh)

* Fine-tuning
  * GLUE
    * [First Char](./scripts/finetune/glue/glue_first-char.sh)
    * [Last Char](./scripts/finetune/glue/glue_last-n-chars.sh)
    * [First $n$ Chars](./scripts/finetune/glue/glue_first-n-chars.sh)
    * [Last $n$ Chars](./scripts/finetune/glue/glue_last-n-chars.sh)
    * [MLM](./scripts/finetune/glue/glue_mlm.sh)

  * SQuAD
    * [First Char](./scripts/finetune/squad/squad_first-char.sh)
    * [Last Char](./scripts/finetune/squad/squad_last-char.sh)
    * [First $n$ Chars](./scripts/finetune/squad/squad_first-n-chars.sh)
    * [Last $n$ Chars](./scripts/finetune/squad/squad_last-n-chars.sh)
    * [MLM](./scripts/finetune/squad/squad_mlm.sh)

  * Dependency Parsing (UD)
    * [First Char](./scripts/finetune/ud/ud_first-char.sh)
    * [Last Char](./scripts/finetune/ud/ud_last-char.sh)
    * [First $n$ Chars](./scripts/finetune/ud/ud_first-n-chars.sh)
    * [Last $n$ Chars](./scripts/finetune/ud/ud_last-n-chars.sh)
    * [MLM](./scripts/finetune/ud/ud_mlm.sh)


### Pretrained Weights
Our pretrained weights are available on the HuggingFace Hub:
* [First Char](https://huggingface.co/hitachi-nlp/roberta-base_first-char_acl2023)
* [Last Char](https://huggingface.co/hitachi-nlp/roberta-base_last-char_acl2023)
* First $n$ Chars
  * [1 Char](https://huggingface.co/hitachi-nlp/roberta-base_first-1-char_acl2023)
  * [2 Chars](https://huggingface.co/hitachi-nlp/roberta-base_first-2-chars_acl2023)
  * [3 Chars](https://huggingface.co/hitachi-nlp/roberta-base_first-3-chars_acl2023)
  * [4 Chars](https://huggingface.co/hitachi-nlp/roberta-base_first-4-chars_acl2023)
  * [5 Chars](https://huggingface.co/hitachi-nlp/roberta-base_first-5-chars_acl2023)
  * [9 Chars](https://huggingface.co/hitachi-nlp/roberta-base_first-9-chars_acl2023)
* Last $n$ Chars
  * [1 Char](https://huggingface.co/hitachi-nlp/roberta-base_last-1-char_acl2023)
  * [2 Chars](https://huggingface.co/hitachi-nlp/roberta-base_last-2-chars_acl2023)
  * [3 Chars](https://huggingface.co/hitachi-nlp/roberta-base_last-3-chars_acl2023)
  * [4 Chars](https://huggingface.co/hitachi-nlp/roberta-base_last-4-chars_acl2023)
  * [5 Chars](https://huggingface.co/hitachi-nlp/roberta-base_last-5-chars_acl2023)
  * [9 Chars](https://huggingface.co/hitachi-nlp/roberta-base_last-9-chars_acl2023)
* [MLM](https://huggingface.co/hitachi-nlp/roberta-base_mlm_acl2023)


## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a> unless specified.
