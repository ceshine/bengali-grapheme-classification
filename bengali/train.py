import os
import logging
from typing import Tuple, Optional

import fire
import numpy as np
import tensorflow as tf
from tf_helper_bot import (
    BaseBot, BaseDistributedBot,
    MovingAverageStatsTrackerCallback, CheckpointCallback,
    CosineDecayWithWarmup, WandbCallback,
    TelegramCallback
)
from tf_helper_bot.utils import prepare_tpu
from tf_helper_bot.optimizers import RAdam, LAMB
from tensorflow.keras.mixed_precision import experimental as amp
import tensorflow_addons as tfa

from .model import get_model
from .dataset import tfrecord_dataset, SPLIT_POINT, IMAGE_SIZE
from .metrics import MacroAveragedRecall


def loss_fn(labels, predictions):
    losses = []
    for i in range(3):
        if isinstance(labels, dict):
            loss_1 = tf.cast(tf.keras.losses.sparse_categorical_crossentropy(
                labels["labels_1"][:, i],
                predictions[:, SPLIT_POINT[i]:SPLIT_POINT[i+1]]
            ), tf.float32)
            loss_2 = tf.cast(tf.keras.losses.sparse_categorical_crossentropy(
                labels["labels_2"][:, i],
                predictions[:, SPLIT_POINT[i]:SPLIT_POINT[i+1]]
            ), tf.float32)
            losses.append(
                tf.reduce_mean(
                    labels["lambd"] * loss_1 + (1 - labels["lambd"]) * loss_2
                ))
        else:
            losses.append(
                tf.cast(tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        labels[:, i],
                        predictions[:, SPLIT_POINT[i]:SPLIT_POINT[i+1]]
                    )
                ), tf.float32)
            )
    return losses[0] * tf.constant(2., dtype=tf.float32) + losses[1] + losses[2]


def main(
    arch: str = "b3", batch_size: int = 8,
    train_folder: str = "data/tfrecords/train/",
    valid_folder: str = "data/tfrecords/valid/",
    grad_accu: int = 1,
    checkpoint_interval: int = 1000,
    steps: int = 10000,
    log_interval: int = 100,
    max_lr: float = 3e-4,
    min_lr: float = 1e-6,
    resize: Optional[Tuple[int, int]] = None,
    mixup_alpha: float = -1,
    cutmix_alpha: float = -1,
    cutmix_ratio: float = 0.5,
    mixed_precision: bool = False,
    output_suffix: str = "",
    weight_decay: float = 0.,
    max_rotate_deg: float = 0.,
    radam: bool = False,
    lookahead_sync: bool = False,
    weights: str = "noisy-student"
):
    strategy, tpu = prepare_tpu(zone=os.environ.get("TPU_ZONE"))
    print("REPLICAS: ", strategy.num_replicas_in_sync)

    if mixed_precision:
        if tpu:
            policy = amp.Policy('mixed_bfloat16')
        else:
            policy = amp.Policy('mixed_float16')
        amp.set_policy(policy)
        print("TPU:", tpu, type(strategy))
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)

    valid_batch_size = batch_size * 2
    if strategy.num_replicas_in_sync == 8:  # single TPU
        valid_batch_size = batch_size * strategy.num_replicas_in_sync
        batch_size = batch_size * strategy.num_replicas_in_sync
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    with strategy.scope():
        model = get_model(
            arch, image_size=IMAGE_SIZE if resize is None else (*resize, 3),
            pretrained=weights
        )
        lr_schedule = CosineDecayWithWarmup(
            initial_learning_rate=min_lr, max_learning_rate=max_lr,
            warmup_steps=int(steps * 0.2),
            decay_steps=steps - int(steps * 0.2),
            alpha=1e-4
        )
        if radam:
            optimizer = tfa.optimizers.RectifiedAdam(
                learning_rate=lr_schedule, epsilon=1e-6)
        else:
            optimizer = tfa.optimizers.AdamW(
                learning_rate=lr_schedule,
                epsilon=1e-6, weight_decay=weight_decay)
        # optimizer = LAMB(learning_rate=lr_schedule,
        #                  epsilon=1e-6, weight_decay=weight_decay)
        if lookahead_sync > 0:
            optimizer = tfa.optimizers.Lookahead(
                optimizer, sync_period=lookahead_sync, slow_step_size=0.5)
        optimizer_name = str(type(optimizer))
        if mixed_precision:
            optimizer = amp.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    print(model.summary())

    train_dataset, train_steps = tfrecord_dataset(
        tf.io.gfile.glob(train_folder + "*"),
        batch_size, is_train=True, strategy=strategy,
        resize=resize, mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_ratio=cutmix_ratio,
        max_rotate_deg=float(max_rotate_deg)
    )
    valid_dataset, valid_steps = tfrecord_dataset(
        tf.io.gfile.glob(valid_folder + "*"),
        valid_batch_size, is_train=False, strategy=strategy, resize=resize
    )

    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir="cache/model_cache/",
        monitor_metric="ma_recall"
    )
    callbacks = (
        MovingAverageStatsTrackerCallback(
            avg_window=int(log_interval * 1.25),
            log_interval=log_interval,
        ),
        checkpoints,
        WandbCallback(
            config={
                "arch": arch,
                "steps": steps,
                "lr": max_lr,
                "resize": resize,
                "batch_size": batch_size,
                "mixup_alpha": mixup_alpha,
                "cutmix_alpha": cutmix_alpha,
                "cutmix_ratio": cutmix_ratio,
                "optimizer": optimizer_name,
                "train_folder": train_folder,
                "valid_folder": valid_folder,
                "output_suffix": output_suffix,
                "weight_decay": weight_decay,
                "max_rotate_deg": max_rotate_deg,
                "lookahead_sync": lookahead_sync,
                "mixed_precision": mixed_precision,
                "radam": radam
            },
            name="Bengali"
        )
        # TelegramCallback(
        #     token="YOUR_TELEGRAM_BOT_TOKEN",
        #     chat_id="YOUR_CHAT_ID", name="Bengali",
        #     report_evals=False
        # )
    )
    metrics = (MacroAveragedRecall(),)
    if tpu:
        train_dist_ds = strategy.experimental_distribute_dataset(
            train_dataset)
        valid_dist_ds = strategy.experimental_distribute_dataset(
            valid_dataset)
        bot = BaseDistributedBot(
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            train_dataset=train_dist_ds,
            valid_dataset=valid_dist_ds,
            steps_per_epoch=train_steps,
            strategy=strategy,
            gradient_accumulation_steps=1,
            callbacks=callbacks,
            metrics=metrics,
            valid_steps=valid_steps,
            mixed_precision=mixed_precision
        )
    else:
        bot = BaseBot(
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            steps_per_epoch=train_steps,
            gradient_accumulation_steps=grad_accu,
            callbacks=callbacks,
            metrics=metrics,
            valid_steps=valid_steps,
            mixed_precision=mixed_precision
        )

    print(f"Steps per epoch: {train_steps} | {valid_steps}")

    bot.train(checkpoint_interval=checkpoint_interval, n_steps=steps)
    bot.model.load_weights(str(checkpoints.best_performers[0][1]))
    checkpoints.remove_checkpoints(keep=0)
    bot.model.save_weights(f"cache/{arch}{output_suffix}.h5")


if __name__ == '__main__':
    fire.Fire(main)
