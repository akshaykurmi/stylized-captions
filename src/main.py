import logging
import os

import argparse
import shutil

from .datasets import PersonalityCaptions, DatasetManager
from .evaluate import generate_captions_for_image, score_on_test_set, human_performance_on_test_set
from .schedules import ExponentialSchedule
from .train import pretrain_generator, pretrain_discriminator, adversarially_train_generator_and_discriminator
from .utils import init_logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--run_id", required=True, type=int)

parser.add_argument("--overwrite_run_results", default=False, action="store_true")
parser.add_argument("--overwrite_cached_dataset", default=False, action="store_true")
parser.add_argument("--run_download_dataset", default=False, action="store_true")
parser.add_argument("--run_cache_dataset", default=False, action="store_true")
parser.add_argument("--run_generator_pretraining", default=False, action="store_true")
parser.add_argument("--run_discriminator_pretraining", default=False, action="store_true")
parser.add_argument("--run_adversarial_training", default=False, action="store_true")

parser.add_argument("--run_human_evaluation", default=False, action="store_true")
parser.add_argument("--run_evaluation", default=False, action="store_true")
parser.add_argument("--checkpoints_to_evaluate", default="")
parser.add_argument("--generate_captions_for_image", default=False, action="store_true")
parser.add_argument("--checkpoint_to_generate_from", default="")

parser.add_argument("--stylize", default=False, action="store_true")

args = parser.parse_args()

args.run_id = f"run_{args.run_id}"
args.base_dir = os.path.dirname(os.path.dirname(__file__))
args.data_dir = os.path.join(args.base_dir, "data", "personality_captions_data")
args.cache_dir = os.path.join(args.data_dir, "cache")
args.results_dir = os.path.join(args.base_dir, "results")
args.run_dir = os.path.join(args.results_dir, args.run_id)
args.checkpoints_dir = os.path.join(args.run_dir, "checkpoints")
args.log_dir = os.path.join(args.run_dir, "logs")

args.seed = 42
args.max_seq_len = 20

args.generator_encoder_units = 2048
args.generator_token_embedding_units = 512
args.generator_style_embedding_units = 64
args.generator_attention_units = 512
args.generator_lstm_units = 512
args.generator_z_units = 256
args.generator_lstm_dropout = 0.5
args.discriminator_token_embedding_units = 512
args.discriminator_style_embedding_units = 64
args.discriminator_lstm_units = 512

args.teacher_forcing_schedule = ExponentialSchedule(0.9999)
args.generator_pretrain_learning_rate = 1e-4
args.generator_pretrain_grad_clipvalue = 5.
args.generator_pretrain_dsa_lambda = 1.0
args.generator_pretrain_batch_size = 64
args.generator_pretrain_epochs = 20
args.generator_pretrain_logging_steps = 20
args.generator_pretrain_validate_steps = 1000
args.generator_pretrain_checkpoint_steps = 500

args.discriminator_pretrain_learning_rate = 1e-4
args.discriminator_pretrain_grad_clipvalue = 5.
args.discriminator_pretrain_batch_size = 64
args.discriminator_pretrain_neg_sample_weight = 0.5
args.discriminator_pretrain_epochs = 10
args.discriminator_pretrain_logging_steps = 20
args.discriminator_pretrain_validate_steps = 1000
args.discriminator_pretrain_checkpoint_steps = 500

args.generator_adversarial_learning_rate = 1e-4
args.generator_adversarial_grad_clipvalue = 5.
args.generator_adversarial_logging_steps = 1
args.generator_adversarial_batch_size = 64
args.generator_adversarial_dsa_lambda = 1.0
args.discriminator_adversarial_learning_rate = 1e-4
args.discriminator_adversarial_grad_clipvalue = 5.
args.discriminator_adversarial_logging_steps = 1
args.discriminator_adversarial_batch_size = 64
args.discriminator_adversarial_neg_sample_weight = 0.5
args.adversarial_rounds = 10000
args.adversarial_validate_rounds = 500
args.adversarial_checkpoint_rounds = 5
args.adversarial_g_steps = 1
args.adversarial_d_steps = 5
args.adversarial_rollout_n = 5
args.adversarial_rollout_update_rate = 1

init_logging(args.log_dir)

personality_captions = PersonalityCaptions(args.data_dir)
dataset_manager = DatasetManager(personality_captions, args.max_seq_len)

if args.run_download_dataset:
    logger.info("***** Downloading Dataset *****")
    personality_captions.download()

if args.run_cache_dataset:
    logger.info("***** Caching dataset as TFRecords *****")
    if args.overwrite_cached_dataset:
        shutil.rmtree(args.cache_dir, ignore_errors=True)
    os.makedirs(args.cache_dir, exist_ok=False)
    dataset_manager.cache_dataset("val", batch_size=32, num_batches_per_shard=80)
    dataset_manager.cache_dataset("test", batch_size=32, num_batches_per_shard=80)
    dataset_manager.cache_dataset("train", batch_size=32, num_batches_per_shard=80)

if args.run_generator_pretraining:
    if args.overwrite_run_results:
        shutil.rmtree(args.run_dir, ignore_errors=True)
    pretrain_generator(args, dataset_manager)

if args.run_discriminator_pretraining:
    if args.overwrite_run_results:
        shutil.rmtree(args.run_dir, ignore_errors=True)
    pretrain_discriminator(args, dataset_manager)

if args.run_adversarial_training:
    if args.overwrite_run_results:
        shutil.rmtree(args.run_dir, ignore_errors=True)
    adversarially_train_generator_and_discriminator(args, dataset_manager)

if args.run_evaluation:
    checkpoint_numbers = [int(c) for c in args.checkpoints_to_evaluate.split(",")]
    score_on_test_set(args, dataset_manager, checkpoint_numbers)

if args.run_human_evaluation:
    human_performance_on_test_set(dataset_manager)

if args.generate_captions_for_image:
    generate_captions_for_image(args, dataset_manager, int(args.checkpoint_to_generate_from))
