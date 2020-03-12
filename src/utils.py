import logging
import os
import random
import shutil

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def init_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), mode='w')
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(log_format)
    fh.setFormatter(log_format)
    logging.basicConfig(level=logging.INFO, handlers=[ch, fh])
    tf.get_logger().setLevel(logging.ERROR)


class MultiCheckpointManager:
    def __init__(self, checkpoints_dir, overwrite_checkpoint_dir, config):
        self.checkpoints = {}
        self.checkpoint_managers = {}
        if overwrite_checkpoint_dir:
            shutil.rmtree(checkpoints_dir)
        for checkpoint_name, objects_to_save in config.items():
            checkpoint = tf.train.Checkpoint(**objects_to_save)
            manager = tf.train.CheckpointManager(checkpoint, os.path.join(checkpoints_dir, checkpoint_name),
                                                 max_to_keep=1, keep_checkpoint_every_n_hours=1,
                                                 checkpoint_name=checkpoint_name)
            self.checkpoints[checkpoint_name] = checkpoint
            self.checkpoint_managers[checkpoint_name] = manager

    def restore_latest(self):
        for checkpoint_name in self.checkpoints.keys():
            self.checkpoints[checkpoint_name].restore(self.checkpoint_managers[checkpoint_name].latest_checkpoint)

    def save(self, checkpoint_number):
        for manager in self.checkpoint_managers.values():
            manager.save(checkpoint_number=checkpoint_number)
