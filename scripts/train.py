"""Train a LayoutLM Model using PyTorch Lightning"""
import click
import lightning.pytorch as pl

from layoutlm_lightning.module import LayoutLMLightningModule, FUNSDFormatDataModule

from logging import getLogger

logger = getLogger(__name__)

@click.command("Train LayoutLM using PyTorch Lightning")
@click.option("--data_dir", help="Directory Containing the data")
@click.option("--accelerator", default="cpu")
def main(data_dir, accelerator):
    logger.info("Loading DataModule")
    
    datamodule = FUNSDFormatDataModule(data_dir=data_dir)

    logger.info("Training Model") 
    trainer = pl.Trainer(accelerator=accelerator, fast_dev_run=True)

    trainer.fit(model=LayoutLMLightningModule(), datamodule=datamodule)

if __name__ == "__main__":
        main()