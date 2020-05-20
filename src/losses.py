import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class GeneratorMLELoss:
    def __call__(self, captions, logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(captions, logits)
        mask = tf.cast(tf.sign(tf.abs(captions)), tf.float32)
        loss *= mask
        loss = tf.reduce_sum(loss, axis=1)
        loss /= tf.reduce_sum(mask, axis=1)
        nll_loss = tf.reduce_mean(loss)
        # dsa_loss = dsa_lambda * tf.reduce_mean((1 - tf.reduce_sum(attention_alphas, axis=1)) ** 2)
        return nll_loss


class PolicyGradientLoss:
    def __call__(self, captions, logits, rewards):
        probabilities = tf.nn.softmax(logits)
        mask = tf.cast(tf.sign(tf.abs(captions)), tf.float32)
        sequence_lengths = tf.reduce_sum(mask, axis=1)
        sequence_lengths = tf.broadcast_to(tf.expand_dims(tf.constant(sequence_lengths), axis=1), rewards.shape)
        rewards = rewards - 0.5 * (1 / sequence_lengths)

        probabilities = tf.reshape(probabilities, shape=(-1, probabilities.shape[-1]))
        captions = tf.reshape(captions, shape=(-1,))
        rewards = tf.reshape(rewards, shape=(-1,))
        mask = tf.reshape(mask, shape=(-1,))
        indices = tf.stack([tf.range(captions.shape[0], dtype=tf.int64), captions], axis=1)
        probabilities = tf.gather_nd(probabilities, indices)

        mask = tf.cast(mask, tf.bool)
        probabilities = probabilities[mask]
        rewards = rewards[mask]
        mean_reward = tf.reduce_mean(rewards)
        rewards -= mean_reward
        rewards /= tf.math.reduce_std(rewards)
        loss = -tf.reduce_sum(tf.math.log(probabilities) * rewards)
        return loss, mean_reward
