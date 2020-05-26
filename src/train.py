import logging
import os
from collections import defaultdict

import tensorflow as tf
from tqdm import tqdm

from .transformer import TransformerGenerator
from .losses import GeneratorMLELoss, PolicyGradientLoss
from .models import Generator, Discriminator
from .utils import set_seed, MultiCheckpointManager

logger = logging.getLogger(__name__)


class MonteCarloRollout:
    def __init__(self, generator, n_rollouts, update_rate):
        self.generator = generator
        self.update_rate = update_rate
        self.n_rollouts = n_rollouts

    def update_weights(self, generator):
        updated_weights = []
        variable_names = [v.name for v in generator.variables]
        for name, self_w, other_w in zip(variable_names, self.generator.get_weights(), generator.get_weights()):
            if "embedding" in name:
                updated_weights.append(other_w)
            else:
                updated_weights.append(self.update_rate * other_w + (1 - self.update_rate) * self_w)
        self.generator.set_weights(updated_weights)

    def calculate_rewards(self, encoder_output, captions, styles, discriminator, tokenizer, training):
        sequence_length = captions.shape[1]
        rewards = []
        for t in range(1, sequence_length):
            initial_sequence = captions[:, :t]
            samples = self.generator.sample(encoder_output, initial_sequence, styles, sequence_length,
                                            mode="stochastic", n_samples=self.n_rollouts, training=training,
                                            sos=tokenizer.start_id, eos=tokenizer.end_id)[0]
            rewards_t = tf.reduce_mean([
                discriminator(encoder_output, s, styles, training=False) for s in samples
            ], axis=0)
            rewards.append(rewards_t)
        rewards.append(discriminator(encoder_output, captions, styles, training=False))
        return tf.squeeze(tf.stack(rewards, axis=1))


# @tf.function
def generator_train_batch_pg(batch, generator, discriminator, optimizer, loss_fn, rollout, tokenizer, max_seq_len):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(generator.trainable_variables)
        encoder_output, styles = batch[0], batch[2]
        batch_size = encoder_output.shape[0]
        initial_sequence = tf.ones((batch_size, 1), dtype=tf.int64) * tokenizer.start_id
        captions, logits = generator.sample(encoder_output, initial_sequence, styles,
                                            sequence_length=max_seq_len, mode="stochastic", n_samples=1,
                                            training=True, sos=tokenizer.start_id, eos=tokenizer.end_id)
        captions, logits = captions[0], logits[0]
        rewards = rollout.calculate_rewards(encoder_output, captions, styles, discriminator, tokenizer, training=True)
        loss, reward = loss_fn(captions, logits, rewards)
        gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss, reward


# @tf.function
def generator_loss_pg(batch, generator, discriminator, loss_fn, rollout, tokenizer, max_seq_len):
    encoder_output, styles = batch[0], batch[2]
    batch_size = encoder_output.shape[0]
    initial_sequence = tf.ones((batch_size, 1), dtype=tf.int64) * tokenizer.start_id
    captions, logits = generator.sample(encoder_output, initial_sequence, styles, sequence_length=max_seq_len,
                                        mode="stochastic", n_samples=1, training=False,
                                        sos=tokenizer.start_id, eos=tokenizer.end_id)
    captions, logits = captions[0], logits[0]
    rewards = rollout.calculate_rewards(encoder_output, captions, styles, discriminator, tokenizer, training=False)
    loss, reward = loss_fn(captions, logits, rewards)
    return loss, reward


@tf.function
def generator_train_batch_mle(batch, generator, loss_fn, optimizer):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(generator.trainable_variables)
        encoder_output, captions, styles = batch[0], batch[1], batch[2]
        # logits, attention_alphas, sort_indices = generator.forward(encoder_output, captions, styles,
        #                                                            teacher_forcing_rate=teacher_forcing_rate,
        #                                                            mode="stochastic", training=True)
        # captions = tf.gather(captions, sort_indices)[:, 1:]
        logits = generator.forward(encoder_output, captions, styles, training=True)
        captions = captions[:, 1:]
        logits = logits[:, :-1, :]
        loss = loss_fn(captions, logits)
        gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss


@tf.function
def generator_loss_mle(batch, generator, loss_fn):
    encoder_output, captions, styles = batch[0], batch[1], batch[2]
    # logits, attention_alphas, sort_indices = generator.forward(encoder_output, captions, styles,
    #                                                            mode="stochastic", teacher_forcing_rate=0,
    #                                                            training=False)
    # captions = tf.gather(captions, sort_indices)[:, 1:]
    logits = generator.forward(encoder_output, captions, styles, training=True)
    captions = captions[:, 1:]
    logits = logits[:, :-1, :]
    loss = loss_fn(captions, logits)
    return loss


@tf.function
def discriminator_train_batch_mle(batches, discriminator, loss_fn, optimizer):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(discriminator.trainable_variables)
        encoder_output = tf.concat([b[0] for b in batches], axis=0)
        labels = tf.concat([b[2] for b in batches], axis=0)
        sample_weight = tf.concat([b[3] for b in batches], axis=0)
        styles = tf.concat([b[4] for b in batches], axis=0)
        captions = [b[1] for b in batches]
        max_caption_length = max([c.shape[1] for c in captions])
        captions = [tf.pad(c, paddings=tf.constant([[0, 0], [0, max_caption_length - c.shape[1]]])) for c in captions]
        captions = tf.concat(captions, axis=0)
        predictions = discriminator(encoder_output, captions, styles, training=True)
        loss = loss_fn(labels, predictions, sample_weight=sample_weight)
        gradients = tape.gradient(loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return loss


@tf.function
def discriminator_loss_mle(batches, discriminator, loss_fn):
    encoder_output = tf.concat([b[0] for b in batches], axis=0)
    labels = tf.concat([b[2] for b in batches], axis=0)
    sample_weight = tf.concat([b[3] for b in batches], axis=0)
    styles = tf.concat([b[4] for b in batches], axis=0)
    captions = [b[1] for b in batches]
    max_caption_length = max([c.shape[1] for c in captions])
    captions = [tf.pad(c, paddings=tf.constant([[0, 0], [0, max_caption_length - c.shape[1]]])) for c in captions]
    captions = tf.concat(captions, axis=0)
    predictions = discriminator(encoder_output, captions, styles, training=False)
    loss = loss_fn(labels, predictions, sample_weight=sample_weight)
    return loss


@tf.function
def generate_fake_captions(true_batch, generator, tokenizer, max_seq_len):
    encoder_output, labels, sample_weights, styles = true_batch[0], true_batch[2], true_batch[3], true_batch[4]
    batch_size = encoder_output.shape[0]
    initial_sequence = tf.ones((batch_size, 1), dtype=tf.int64) * tokenizer.start_id
    captions = generator.sample(encoder_output, initial_sequence, styles, sequence_length=max_seq_len,
                                mode="stochastic", n_samples=1, training=False,
                                sos=tokenizer.start_id, eos=tokenizer.end_id)[0][0]
    return encoder_output, captions, labels, sample_weights, styles


def pretrain_generator(args, dataset_manager):
    logger.info("***** Pretraining Generator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    # generator = Generator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
    #                       style_vocab_size=dataset_manager.style_encoder.num_classes,
    #                       style_embedding_units=args.generator_style_embedding_units,
    #                       token_embedding_units=args.generator_token_embedding_units,
    #                       lstm_units=args.generator_lstm_units, lstm_dropout=args.generator_lstm_dropout,
    #                       attention_units=args.generator_attention_units, encoder_units=args.generator_encoder_units,
    #                       z_units=args.generator_z_units, stylize=args.stylize)
    generator = TransformerGenerator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                                     style_vocab_size=dataset_manager.style_encoder.num_classes,
                                     model_dim=512, style_dim=64, pffn_dim=2048, z_dim=512,
                                     encoder_blocks=2, decoder_blocks=6, num_attention_heads=8, max_pe=64,
                                     dropout=0.1, stylize=True)

    loss_fn = GeneratorMLELoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_pretrain_learning_rate,
                                         clipvalue=args.generator_pretrain_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "generator": {"generator": generator},
        "generator_pretrain_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_dataset = dataset_manager.load_generator_dataset(
        "train", batch_size=args.generator_pretrain_batch_size, repeat=args.generator_pretrain_epochs)
    val_dataset = dataset_manager.load_generator_dataset(
        "val", batch_size=args.generator_pretrain_batch_size, repeat=1)

    for train_batch in tqdm(train_dataset, desc="Batch", unit="batch"):
        global_step.assign_add(1)
        loss = generator_train_batch_mle(train_batch, generator, loss_fn, optimizer)
        if global_step % args.generator_pretrain_logging_steps == 0:
            with train_summary_writer.as_default(), tf.name_scope("generator_pretraining"):
                tf.summary.scalar("mle_loss", loss, step=global_step)
                # tf.summary.scalar("nll_loss", nll_loss, step=global_step)
                # tf.summary.scalar("dsa_loss", dsa_loss, step=global_step)
                # tf.summary.scalar("teacher_forcing_rate", teacher_forcing_rate, step=global_step)
        if global_step % args.generator_pretrain_validate_steps == 0:
            logger.info("-- Calculating validation loss")
            losses = [generator_loss_mle(val_batch, generator, loss_fn) for val_batch in val_dataset]
            with val_summary_writer.as_default(), tf.name_scope("generator_pretraining"):
                tf.summary.scalar("mle_loss", tf.reduce_mean(losses), step=global_step)
                # tf.summary.scalar("nll_loss", tf.reduce_mean(losses[1]), step=global_step)
                # tf.summary.scalar("dsa_loss", tf.reduce_mean(losses[2]), step=global_step)
        if global_step % args.generator_pretrain_checkpoint_steps == 0:
            checkpoint_manager.save(["generator", "generator_pretrain_params"])

    checkpoint_manager.save(["generator", "generator_pretrain_params"])
    logger.info("***** Pretraining Generator - Ended *****")


def pretrain_discriminator(args, dataset_manager):
    logger.info("***** Pretraining Discriminator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    generator = Generator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                          style_vocab_size=dataset_manager.style_encoder.num_classes,
                          style_embedding_units=args.generator_style_embedding_units,
                          token_embedding_units=args.generator_token_embedding_units,
                          lstm_units=args.generator_lstm_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=args.generator_encoder_units,
                          z_units=args.generator_z_units, stylize=args.stylize)
    discriminator = Discriminator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                                  style_vocab_size=dataset_manager.style_encoder.num_classes,
                                  token_embedding_units=args.discriminator_token_embedding_units,
                                  style_embedding_units=args.discriminator_style_embedding_units,
                                  lstm_units=args.discriminator_lstm_units, stylize=args.stylize)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_pretrain_learning_rate,
                                         clipvalue=args.discriminator_pretrain_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "generator": {"generator": generator},
        "discriminator": {"discriminator": discriminator},
        "discriminator_pretrain_params": {"optimizer": optimizer, "global_step": global_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    train_dataset, val_dataset = {}, {}
    for s, d, r in [("train", train_dataset, args.discriminator_pretrain_epochs), ("val", val_dataset, 1)]:
        d["true"] = dataset_manager.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_pretrain_batch_size, repeat=r,
            label=1, randomize_captions=False, sample_weight=1)
        d["fake"] = dataset_manager.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_pretrain_batch_size, repeat=r,
            label=0, randomize_captions=False, sample_weight=args.discriminator_pretrain_neg_sample_weight)
        d["shuffled"] = dataset_manager.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_pretrain_batch_size, repeat=r,
            label=0, randomize_captions=True, sample_weight=args.discriminator_pretrain_neg_sample_weight)

    for true_batch, fake_batch, shuffled_batch in tqdm(zip(train_dataset["true"], train_dataset["fake"],
                                                           train_dataset["shuffled"]), desc="Batch", unit="batch"):
        global_step.assign_add(1)
        fake_batch = generate_fake_captions(fake_batch, generator, dataset_manager.tokenizer, args.max_seq_len)
        loss = discriminator_train_batch_mle((true_batch, fake_batch, shuffled_batch), discriminator, loss_fn,
                                             optimizer)
        if global_step % args.discriminator_pretrain_logging_steps == 0:
            with train_summary_writer.as_default(), tf.name_scope("discriminator_pretraining"):
                tf.summary.scalar("nll_loss", loss, step=global_step)
        if global_step % args.discriminator_pretrain_validate_steps == 0:
            logger.info("-- Calculating validation loss")
            losses = []
            for val_true_batch, val_fake_batch, val_shuffled_batch in zip(val_dataset["true"], val_dataset["fake"],
                                                                          val_dataset["shuffled"]):
                val_fake_batch = generate_fake_captions(val_fake_batch, generator, dataset_manager.tokenizer,
                                                        args.max_seq_len)
                losses.append(discriminator_loss_mle((val_true_batch, val_fake_batch, val_shuffled_batch),
                                                     discriminator, loss_fn))
            with val_summary_writer.as_default(), tf.name_scope("discriminator_pretraining"):
                tf.summary.scalar("nll_loss", tf.reduce_mean(losses), step=global_step)
        if global_step % args.discriminator_pretrain_checkpoint_steps == 0:
            checkpoint_manager.save(["discriminator", "discriminator_pretrain_params"])

    checkpoint_manager.save(["discriminator", "discriminator_pretrain_params"])
    logger.info("***** Pretraining Discriminator - Ended *****")


def adversarially_train_generator_and_discriminator(args, dataset_manager):
    logger.info("***** Adversarially training Generator & Discriminator - Started *****")

    set_seed(args.seed)

    logger.info("-- Initializing")
    generator = Generator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                          style_vocab_size=dataset_manager.style_encoder.num_classes,
                          style_embedding_units=args.generator_style_embedding_units,
                          token_embedding_units=args.generator_token_embedding_units,
                          lstm_units=args.generator_lstm_units, lstm_dropout=args.generator_lstm_dropout,
                          attention_units=args.generator_attention_units, encoder_units=args.generator_encoder_units,
                          z_units=args.generator_z_units, stylize=args.stylize)
    generator_mc = Generator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                             style_vocab_size=dataset_manager.style_encoder.num_classes,
                             style_embedding_units=args.generator_style_embedding_units,
                             token_embedding_units=args.generator_token_embedding_units,
                             lstm_units=args.generator_lstm_units, lstm_dropout=args.generator_lstm_dropout,
                             attention_units=args.generator_attention_units, encoder_units=args.generator_encoder_units,
                             z_units=args.generator_z_units, stylize=args.stylize)
    discriminator = Discriminator(token_vocab_size=dataset_manager.tokenizer.vocab_size,
                                  style_vocab_size=dataset_manager.style_encoder.num_classes,
                                  token_embedding_units=args.discriminator_token_embedding_units,
                                  style_embedding_units=args.discriminator_style_embedding_units,
                                  lstm_units=args.discriminator_lstm_units, stylize=args.stylize)
    rollout = MonteCarloRollout(generator_mc, args.adversarial_rollout_n, args.adversarial_rollout_update_rate)
    rollout.update_weights(generator)

    generator_loss_fn_pg = PolicyGradientLoss()
    generator_loss_fn_mle = GeneratorMLELoss()
    discriminator_loss_fn = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.generator_adversarial_learning_rate,
                                                   clipvalue=args.generator_adversarial_grad_clipvalue)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.discriminator_adversarial_learning_rate,
                                                       clipvalue=args.discriminator_adversarial_grad_clipvalue)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "val"))

    generator_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    discriminator_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint_manager = MultiCheckpointManager(args.checkpoints_dir, {
        "generator": {"generator": generator},
        "discriminator": {"discriminator": discriminator},
        "adversarial_params": {"generator_optimizer": generator_optimizer,
                               "discriminator_optimizer": discriminator_optimizer,
                               "generator_step": generator_step, "discriminator_step": discriminator_step}
    })
    checkpoint_manager.restore_latest()

    logger.info("-- Loading training and validation sets")
    generator_train_dataset = iter(dataset_manager.load_generator_dataset(
        "train", batch_size=args.generator_adversarial_batch_size, repeat=-1))
    generator_val_dataset = dataset_manager.load_generator_dataset(
        "val", batch_size=args.generator_adversarial_batch_size, repeat=1)
    discriminator_train_dataset, discriminator_val_dataset = {}, {}
    for s, d, r in [("train", discriminator_train_dataset, -1), ("val", discriminator_val_dataset, 1)]:
        d["true"] = dataset_manager.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_adversarial_batch_size, repeat=r,
            label=1, randomize_captions=False, sample_weight=1)
        d["fake"] = dataset_manager.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_adversarial_batch_size, repeat=r,
            label=0, randomize_captions=False, sample_weight=args.discriminator_adversarial_neg_sample_weight)
        d["shuffled"] = dataset_manager.load_discriminator_dataset(
            split=s, batch_size=args.discriminator_adversarial_batch_size, repeat=r,
            label=0, randomize_captions=True, sample_weight=args.discriminator_adversarial_neg_sample_weight)
    discriminator_train_dataset = {k: iter(v) for k, v in discriminator_train_dataset.items()}

    for round_ in range(1, args.adversarial_rounds):
        logger.info(f"-- Round: {round_}/{args.adversarial_rounds}")

        for _ in tqdm(range(args.adversarial_g_steps), desc="Training Generator", unit="batch"):
            generator_step.assign_add(1)
            train_batch = next(generator_train_dataset)
            pg_loss, pg_reward = generator_train_batch_pg(train_batch, generator, discriminator, generator_optimizer,
                                                          generator_loss_fn_pg, rollout, dataset_manager.tokenizer,
                                                          args.max_seq_len)
            if generator_step % args.generator_adversarial_logging_steps == 0:
                mle_loss = generator_loss_mle(train_batch, generator, generator_loss_fn_mle,
                                              args.generator_adversarial_dsa_lambda)
                with train_summary_writer.as_default(), tf.name_scope("generator_adversarial_training"):
                    tf.summary.scalar("policy_gradient_loss", pg_loss, step=generator_step)
                    tf.summary.scalar("policy_gradient_reward", pg_reward, step=generator_step)
                    tf.summary.scalar("mle_loss", mle_loss[0], step=generator_step)
                    tf.summary.scalar("nll_loss", mle_loss[1], step=generator_step)
                    tf.summary.scalar("dsa_loss", mle_loss[2], step=generator_step)

        for _ in tqdm(range(args.adversarial_d_steps), desc="Training Discriminator", unit="batch"):
            discriminator_step.assign_add(1)
            true_batch = next(discriminator_train_dataset["true"])
            fake_batch = next(discriminator_train_dataset["fake"])
            shuffled_batch = next(discriminator_train_dataset["shuffled"])
            fake_batch = generate_fake_captions(fake_batch, generator, dataset_manager.tokenizer,
                                                args.max_seq_len)
            loss = discriminator_train_batch_mle((true_batch, fake_batch, shuffled_batch), discriminator,
                                                 discriminator_loss_fn, discriminator_optimizer)
            if discriminator_step % args.discriminator_adversarial_logging_steps == 0:
                with train_summary_writer.as_default(), tf.name_scope("discriminator_adversarial_training"):
                    tf.summary.scalar("nll_loss", loss, step=discriminator_step)

        rollout.update_weights(generator)

        if round_ % args.adversarial_checkpoint_rounds:
            checkpoint_manager.save(["generator", "discriminator", "adversarial_params"])

        if round_ % args.adversarial_validate_rounds == 0:
            logger.info("-- Calculating generator validation loss")
            generator_losses = defaultdict(list)
            for val_batch in generator_val_dataset:
                generator_losses["pg"].append(
                    generator_loss_pg(val_batch, generator, discriminator, generator_loss_fn_pg, rollout,
                                      dataset_manager.tokenizer, args.max_seq_len)
                )
                generator_losses["mle"].append(generator_loss_mle(val_batch, generator, generator_loss_fn_mle,
                                                                  args.generator_adversarial_dsa_lambda))
            generator_losses["pg"] = list(zip(*generator_losses["pg"]))
            generator_losses["mle"] = list(zip(*generator_losses["mle"]))
            with val_summary_writer.as_default(), tf.name_scope("generator_adversarial_training"):
                tf.summary.scalar("policy_gradient_loss", tf.reduce_mean(generator_losses["pg"][0]),
                                  step=generator_step)
                tf.summary.scalar("policy_gradient_reward", tf.reduce_mean(generator_losses["pg"][1]),
                                  step=generator_step)
                tf.summary.scalar("mle_loss", tf.reduce_mean(generator_losses["mle"][0]), step=generator_step)
                tf.summary.scalar("nll_loss", tf.reduce_mean(generator_losses["mle"][1]), step=generator_step)
                tf.summary.scalar("dsa_loss", tf.reduce_mean(generator_losses["mle"][2]), step=generator_step)

            logger.info("-- Calculating discriminator validation loss")
            discriminator_losses = []
            for val_true_batch, val_fake_batch, val_shuffled_batch in zip(discriminator_val_dataset["true"],
                                                                          discriminator_val_dataset["fake"],
                                                                          discriminator_val_dataset["shuffled"]):
                val_fake_batch = generate_fake_captions(val_fake_batch, generator, dataset_manager.tokenizer,
                                                        args.max_seq_len)
                discriminator_losses.append(discriminator_loss_mle((val_true_batch, val_fake_batch, val_shuffled_batch),
                                                                   discriminator, discriminator_loss_fn))
            with val_summary_writer.as_default(), tf.name_scope("discriminator_adversarial_training"):
                tf.summary.scalar("nll_loss", tf.reduce_mean(discriminator_losses), step=discriminator_step)

    checkpoint_manager.save(["generator", "discriminator", "adversarial_params"])
    logger.info("***** Adversarially training Generator & Discriminator - Ended *****")
