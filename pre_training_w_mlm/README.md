Pretraining is not implemented with HuggingFace Transformer.
https://discuss.huggingface.co/t/example-of-how-to-pretrain-t5/4129/11
https://discuss.huggingface.co/t/training-t5-on-mlm-task-from-scratch/12617
But it is available through a flax t5 script

run_t5_mlm
https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#t5-like-span-masked-language-modeling

QUestions
https://discuss.huggingface.co/t/example-of-how-to-pretrain-t5/4129/11

Create configuration
Next, we create the model's configuration file. This is as simple as loading and storing **google/t5-v1_1-base** in the local model folder:

from transformers import T5Config

config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./norwegian-t5-base")
Great, we have set up our model repository. During training, we will automatically push the training logs and model weights to the repo.

Train model
Next we can run the example script to pretrain the model:

python run_t5_mlm_flax.py \
	--output_dir="./norwegian-t5-base" \
	--model_type="t5" \
	--config_name="./norwegian-t5-base" \
	--tokenizer_name="./norwegian-t5-base" \
	--dataset_name="oscar" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \ # Need to make sure save_steps < batch steps 
	--eval_steps="2500" \
	--push_to_hub
Training should converge at a loss and accuracy of 2.36 and 57.0 respectively after 3 epochs on a single TPUv3-8. This should take around 4.5 hours. Training statistics can be accessed on directly on the 🤗 hub
