import lightning.pytorch as pl

from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from transformers import AdamW

from pathlib import Path
from layoutlm_lightning.dataset import FunsdDataset

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
            mode="train"
        )
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=2)
    
    def train_dataloader(self):
        return self.get_dataloader(mode="train")
    
    def val_dataloader(self):
        return self.get_dataloader(mode="val")

class LayoutLMLightningModule(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=7)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        input_ids = batch[0]
        bbox = batch[4]
        attention_mask = batch[1]
        token_type_ids = batch[2]
        labels = batch[3]

      # forward pass
        outputs = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=labels)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())
        return optimizer