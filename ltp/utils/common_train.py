import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def common_train(args, metric, model_class, build_method, task: str, **model_kwargs):
    pl.seed_everything(args.seed)

    early_stop_callback = EarlyStopping(
        monitor=metric, min_delta=1e-5, patience=3, verbose=False, mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=metric, save_top_k=1, verbose=True, mode='max', save_last=True
    )
    model = model_class(args, **model_kwargs)
    build_method(model)
    this_time = time.strftime("%m-%d_%H:%M:%S", time.localtime())
    logger = loggers.TensorBoardLogger(save_dir='lightning_logs', name=f'{task}_{this_time}')
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[early_stop_callback],
        checkpoint_callback=checkpoint_callback
    )
    # Ready to train with new learning rate
    trainer.fit(model)
    trainer.test()
