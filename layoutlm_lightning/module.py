import collections
import logging
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from transformers import AdamW

from pathlib import Path
from layoutlm_lightning.dataset import FunsdDataset

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


logger = logging.getLogger("lightning.pytorch")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class FUNSDFormatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.prepare_labels()
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    
    def get_args(self, mode="train"):
        args = {
            'local_rank': -1,
            'data_dir': str(self.data_dir),
            'model_name_or_path':'microsoft/layoutlm-base-uncased',
            'max_seq_length': 512,
            'model_type': 'layoutlm',
            'overwrite_cache' : False,
        }

        return AttrDict(args)
    
    def prepare_labels(self):
        labels_path = self.data_dir / "labels.txt" 

        with open(labels_path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        
        self.labels = labels
        self.num_labels = len(labels)
        self.label_map = {i: label for i, label in enumerate(labels)}
        self.pad_token_label_id = CrossEntropyLoss().ignore_index 
        return self

    def get_dataloader(self, mode="train"):
        dataset = FunsdDataset(
            self.get_args(),
            self.tokenizer,
            labels=self.labels,
            pad_token_label_id=self.pad_token_label_id,
            mode=mode
        )
        if mode == "train":
            sampler = RandomSampler(dataset)
        elif mode == "val":
            sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=4)
    
    def train_dataloader(self):
        return self.get_dataloader(mode="train")
    
    def val_dataloader(self):
        return self.get_dataloader(mode="val")

class LayoutLMLightningModule(pl.LightningModule):
    
    def __init__(self, num_labels, label_map, pad_label_token_idx=-100):
        super().__init__()
        #TODO parameterize labels
        self.model = LayoutLMForTokenClassification.from_pretrained(
            "microsoft/layoutlm-base-uncased",
            num_labels=num_labels,
        )

        self.label_map = label_map
        # for accumulating all validtion step outputs
        self.validation_step_outputs = {
            "preds":[],
            "labels":[],
        }

    def generate_pred_from_label_map(self, preds, labels):

        generated_labels = [[] for _ in range(labels.shape[0])]
        generated_preds = [[] for _ in range(labels.shape[0])]

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    generated_labels[i].append(self.label_map[labels[i][j]])
                    generated_preds[i].append(self.label_map[preds[i][j]])

        return generated_preds, generated_labels

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input_ids = batch[0]
        attention_mask = batch[1]
        token_type_ids = batch[2]
        labels = batch[3]
        bbox = batch[4]

      # forward pass
        outputs = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=labels)
        loss = outputs.loss
        self.logger.log_metrics({
            "training_loss": loss.item()
        }, step=self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch[0]
        attention_mask = batch[1]
        token_type_ids = batch[2]
        labels = batch[3]
        bbox = batch[4]

        outputs = self.model(
            input_ids=input_ids, bbox=bbox, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,labels=labels
        )
        loss = outputs.loss
        preds = outputs.logits.detach().cpu()
        preds = preds.argmax(axis=2)

        # Generate prediction as IOB and original length of labels
        generated_preds, generated_labels = self.generate_pred_from_label_map(
            preds.numpy(), labels.detach().cpu().numpy()
        )
        self.validation_step_outputs["preds"].append(generated_preds)
        self.validation_step_outputs["labels"].append(generated_labels)

        self.logger.log_metrics({
            "val_loss": loss.item()
        }, step=self.global_step)
        return preds
    
    def on_validation_epoch_end(self) -> None:
        # all_preds = torch.stack(self.validation_step_outputs["preds"])
        # all_labels = torch.sta
        y_true = self.validation_step_outputs["labels"][0]
        y_pred = self.validation_step_outputs["preds"][0]

        val_classification_report = classification_report(
            y_true, y_pred, output_dict=True
        )

        # log the classification report for the epoch
        flattened_preds = flatten(
            val_classification_report
        )
        self.logger.log_metrics(
            flattened_preds, step=self.global_step
        )
        logger.info(classification_report(y_true, y_pred))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())
        return optimizer