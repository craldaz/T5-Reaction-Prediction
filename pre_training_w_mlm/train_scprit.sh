#!/usr/bin/env bash
python run_t5_mlm_flax.py \
	--output_dir="./model" \
	--model_type="t5" \
	--config_name="./" \
	--tokenizer_name="./" \
	--train_file="./all_smi_i_got.csv" \
	--max_seq_length="256" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="1000" \
	--num_train_epochs="20" \
	--eval_steps="2500"  
