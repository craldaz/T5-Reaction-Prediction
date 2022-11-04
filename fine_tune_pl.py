# Import packages
import argparse
import csv
import logging
import os
from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from transformers import (AdamW, AutoTokenizer,
                          T5ForConditionalGeneration,
                          get_linear_schedule_with_warmup)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()

        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path)
        self.targets = []
        self.predictions = []
        self.ranks = []

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        loss_detached = loss.detach()
        self.log('training_loss', loss_detached, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {"train_loss": loss_detached}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_training_loss', avg_train_loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
       
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0].detach()

        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {"val_loss": loss}
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('avg_val_loss', avg_loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)


    def generate_prediction_sentences(self, batch):
        '''
        logic for generating outputs to bleu
        '''

        prediction_ids = self.model.generate(batch["source_ids"],
                                             max_length=300,
                                             num_beams=self.hparams.num_beams,
                                             num_return_sequences=self.hparams.num_return_sequences,
                                             early_stopping=True,
                                             )
        prediction_sentences = self.tokenizer.batch_decode(
            prediction_ids,
            skip_special_tokens=True)
        
        return prediction_sentences

    def generate_target_sentences(self, batch):
        '''
        logic for generating inputs to bleu
        '''
        labels = batch["target_ids"]
        target_sentences = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        return target_sentences
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
       raise NotImplementedError

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=1)
        t_total = (
            (len(dataloader.dataset) //
             (self.hparams.train_batch_size * max(1, self.hparams.devices)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=1)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "val_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))


@dataclass(frozen=True)
class InputExample:
    """
    Args:

    """
    precursors: str
    products: str


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter='\t'))

    def _create_examples(self, lines: List[List[str]], type: str):
        # print(lines[0])

        examples = [
            InputExample(
                precursors=line[0],
                products=line[1]
            )
            for line in lines
        ]
        return examples

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train_df.csv")), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val_df.csv")), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test_df.csv")), "test")


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, tokenizer, data_dir, type_path,  max_len=512, mask_percent=0.5
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.proc = DataProcessor()
        self._build()

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        # might need to squeeze
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        elif self.type_path == "val":
            examples = self.proc.get_dev_examples(self.data_dir)
        else:
            examples = self.proc.get_test_examples(self.data_dir)

        for example in tqdm(examples):
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.precursors
        target = example.products
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=128, padding='max_length', truncation=True, return_tensors="pt"
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=75, padding='max_length', truncation=True, return_tensors="pt"
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


def get_dataset(tokenizer, type_path, args):
    return YourDataSetClass(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path=type_path,
        max_len=args.max_seq_length)


if __name__ == "__main__":
    output_dir = f"outputs-molt5-small-no-spaces-pl"
    args_dict = dict(
        output_dir=output_dir,  # path to save the checkpoints
        log_dir=output_dir,
        data_dir="USPTO_480k_no_spaces",
        model_name_or_path="laituan245/molt5-small",
        tokenizer_name_or_path="laituan245/molt5-small",
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.05,
        adam_epsilon=1e-8,
        warmup_steps=16000,
        #strategy='ddp',
        train_batch_size=16,
        eval_batch_size=4,
        num_beams=10,
        num_return_sequences=5,
        num_train_epochs=20,
        # val_check_interval=12_500,
        check_val_every_n_epoch=1,
        # eval_delay=1,
        # val_check_interval=50,
        gradient_accumulation_steps=2,
        devices=1,
        accelerator='gpu',
        early_stop_callback=True,
        fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
        # opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        max_grad_norm=1.0,
        seed=42,
        cp_filename='latest'
    )

    args = argparse.Namespace(**args_dict)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir, monitor="avg_val_loss", mode="min", save_top_k=1,
        filename=args.cp_filename
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        # gpus=args.n_gpu,
        devices=args.devices,
        accelerator=args.accelerator,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        # amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        # val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        # eval_delay=args.eval_delay,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir)


    model = T5FineTuner(args)
    trainer = pl.Trainer(
        callbacks=[
            LoggingCallback(),
            checkpoint_callback,
            EarlyStopping(monitor="avg_val_loss", mode='min', min_delta=0.001, patience=3, verbose=True)],
        logger=tb_logger,
        **train_params,
    )
    trainer.fit(model)

print('Done')
