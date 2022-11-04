# Import packages
import argparse
import csv
import logging
import os
from dataclasses import dataclass
from typing import List
from pathlib import Path

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


from fine_tune_pl import T5FineTuner, get_dataset


class Tester(T5FineTuner):
    
    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0].detach()

        # target_sentences = self.generate_target_sentences(batch)
        target_sentences = self.tokenizer.batch_decode(
            batch["target_ids"], skip_special_tokens=True)

        self.targets += [[self.tokenizer.tokenize(tg)] for tg in target_sentences]

        # prediction_sentences = self.generate_prediction_sentences(batch)
        prediction_ids = self.model.generate(batch["source_ids"],
                                             max_length=300,
                                             do_sample=True,
                                             top_k=20,
                                             top_p=0.95,
                                             num_return_sequences=self.hparams['num_return_sequences']
                                             )
        prediction_sentences = self.tokenizer.batch_decode(
            prediction_ids,
            skip_special_tokens=True)

        top_1 = []
        len_batch = len(batch["source_ids"])
        num_outputs = len_batch * self.hparams.num_return_sequences
        for i in range(num_outputs):
            if i % self.hparams.num_return_sequences == 0:
                top_1.append(prediction_sentences[i])
        # print(prediction_sentences)
        # print(len(prediction_sentences))
        self.predictions += [self.tokenizer.tokenize(p) for p in top_1]

        def get_prediction_rank(ps):
            for j, p in enumerate(ps):
                if target == p:
                    return j+1
            return 0


        # Top K Accuracy
        n = self.hparams.num_return_sequences
        for i, target in zip(range(len_batch), target_sentences):
            ps = prediction_sentences[i*n: (i*n)+n]
            rank = get_prediction_rank(ps)
            self.ranks.append(rank)

        self.log('test_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {"test_loss": loss}
        return {"test_loss": loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        bleu = bleu_score(self.predictions, self.targets)
        self.predictions = []
        self.targets = []

        correct = 0
        top_k_dict = {}
        total = len(self.ranks)
        for i in range(1, self.hparams.num_return_sequences+1):
            correct += sum(x==i for x in self.ranks)
            top_k_dict[i] = correct/total * 100
            # print(f'Top-{i} {top_k_dict[i]}')
        self.ranks = []

        self.log('avg_test_loss', avg_loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('bleu', bleu, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('top-1', top_k_dict[1], on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('top-2', top_k_dict[2], on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('top-3', top_k_dict[3], on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('top-4', top_k_dict[4], on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('top-5', top_k_dict[5], on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    
folder = Path('/home/codyaldaz/NLU/molt5/outputs-molt5-base-no-spaces-pl')
save_folder = folder / 'top-p'
ckpt_file_path = folder / 'latest.ckpt'
tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_folder)

model = Tester.load_from_checkpoint(checkpoint_path=ckpt_file_path)
args_dict = {'data_dir': 'USPTO_480k_no_spaces/',
                'max_seq_length': model.hparams['max_seq_length']}
args = argparse.Namespace(**args_dict)
test_dataset = get_dataset(tokenizer=model.tokenizer, type_path='test', args=args)
test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=1)
trainer = pl.Trainer(accelerator='gpu', devices=1, logger=tb_logger)
trainer.test(model=model, dataloaders=test_dataloader)
