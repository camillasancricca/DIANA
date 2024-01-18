import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary
from .deepant import AnomalyDetector, DataModule, TrafficDataset, DeepAnt

SEQ_LEN = 10


class deepant_support:

    def __init__(self):
        self.model = DeepAnt(SEQ_LEN, 1)
        self.anomaly_detector = AnomalyDetector(self.model)
        self.mc = ModelCheckpoint(
            dirpath='checkpoints',
            save_last=True,
            save_top_k=1,
            verbose=True,
            monitor='train_loss',
            mode='min'
        )
        self.trainer = pl.Trainer(max_epochs=30,
                                  accelerator="auto",
                                  devices=1,
                                  callbacks=[self.mc],
                                  # progress_bar_refresh_rate=30,
                                  # fast_dev_run=True,
                                  # overfit_batches=1
                                  )

        self.mc.CHECKPOINT_NAME_LAST = f'DeepAnt-best-checkpoint'

    def fit(self, df):
        dm = DataModule(df, SEQ_LEN)
        self.trainer.fit(self.anomaly_detector, dm)

    def predict(self, df):
        dataset = TrafficDataset(df, SEQ_LEN)
        target_idx = dataset.timestamp
        dm = DataModule(df, SEQ_LEN)
        self.anomaly_detector = AnomalyDetector.load_from_checkpoint('checkpoints/DeepAnt-best-checkpoint.ckpt',
                                                                     model=self.model)
        output = self.trainer.predict(self.anomaly_detector, dm)
        preds_losses = pd.Series(torch.tensor([item[1] for item in output]).numpy(), index=target_idx)
        return preds_losses