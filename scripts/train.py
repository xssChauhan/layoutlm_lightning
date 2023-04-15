"""Train a LayoutLM Model using PyTorch Lightning"""
import click
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

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
    mlflow_logger = MLFlowLogger()
    trainer = pl.Trainer(accelerator=accelerator, max_epochs=5, logger=mlflow_logger)

    trainer.fit(
          model=LayoutLMLightningModule(
                num_labels=datamodule.num_labels,
                label_map=datamodule.label_map
            ),
            datamodule=datamodule
        )

if __name__ == "__main__":
        main()