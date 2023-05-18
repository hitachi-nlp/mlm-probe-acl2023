#!/bin/bash

num_steps=500000
task_name=1-chars

for seed in 42 22 29 97 96
	do
		# mlm
		python -u -m supar.cmds.biaffine_dep train -b -d 0 -c ./scripts/finetune/ud/config/${task_name}.ini -p ${task_name}_${num_steps}_$seed \
			--seed=$seed \
			--train /path/to/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-train.conllu \
			--dev /path/to/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-dev.conllu \
			--test /path/to/ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-test.conllu \
			--encoder bert \
			--bert ./weights/roberta-base_${task_name}_2e-4_42/checkpoint-${num_steps}
	done
