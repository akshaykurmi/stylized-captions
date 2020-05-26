import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.image_encoders import InceptionResNetEncoder


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.depth = self.model_dim // self.num_heads
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.dense = tf.keras.layers.Dense(model_dim)

    def call(self, v, k, q, mask):
        batch_size = q.shape[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)
        scaled_attention, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concatenated_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        output = self.dense(concatenated_attention)
        return output, attention_weights

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def _scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(k.shape[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, model_dim, pffn_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(pffn_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(model_dim, activation="linear")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class PositionalEncoding:
    def __init__(self, positions, model_dim):
        self.encoding = self._compute_encoding(positions, model_dim)

    @staticmethod
    def _compute_encoding(positions, model_dim):
        positions = np.arange(positions)[:, np.newaxis]
        indices = np.arange(model_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (indices // 2)) / np.float32(model_dim))
        angle_rads = positions * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        encoding = angle_rads[np.newaxis, ...]
        return tf.cast(encoding, dtype=tf.float32)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, pffn_dim, num_heads, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.pffn = PointWiseFeedForwardNetwork(model_dim, pffn_dim)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training):
        out1, _ = self.mha(x, x, x, mask)
        out1 = self.dropout1(out1, training=training)
        out1 = self.norm1(x + out1)
        out2 = self.pffn(out1)
        out2 = self.dropout2(out2, training=training)
        out2 = self.norm2(out1 + out2)
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, blocks, model_dim, style_dim, pffn_dim, num_heads, style_vocab_size, dropout, stylize):
        super().__init__()
        self.stylize = stylize
        self.initial_layer = PointWiseFeedForwardNetwork(model_dim, pffn_dim)
        self.style_embedding = tf.keras.layers.Embedding(style_vocab_size, style_dim)
        self.encoder_blocks = [EncoderBlock(model_dim, pffn_dim, num_heads, dropout) for _ in range(blocks)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, image_feature, style, encoder_padding_mask, training):
        if self.stylize:
            s = self.style_embedding(style)
            s = tf.repeat(tf.expand_dims(s, axis=1), repeats=image_feature.shape[1], axis=1)
            x = tf.concat([image_feature, s], axis=-1)
        else:
            x = image_feature
        x = self.initial_layer(x)
        x = self.dropout(x, training=training)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, encoder_padding_mask, training)
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, pffn_dim, num_heads, dropout):
        super().__init__()
        self.mha1 = MultiHeadAttention(model_dim, num_heads)
        self.mha2 = MultiHeadAttention(model_dim, num_heads)
        self.pffn = PointWiseFeedForwardNetwork(model_dim, pffn_dim)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, encoder_output, padding_mask, look_ahead_mask, training):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(encoder_output, encoder_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(attn2 + out1)
        pffn_output = self.pffn(out2)
        pffn_output = self.dropout3(pffn_output, training=training)
        out3 = self.norm3(pffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, blocks, model_dim, pffn_dim, num_heads, token_vocab_size, dropout, max_pe):
        super().__init__()
        self.model_dim = model_dim
        self.token_embedding = tf.keras.layers.Embedding(token_vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(max_pe, model_dim)
        self.decoder_blocks = [DecoderBlock(model_dim, pffn_dim, num_heads, dropout) for _ in range(blocks)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, encoder_output, caption, padding_mask, look_ahead_mask, training):
        seq_len = caption.shape[1]
        x = self.token_embedding(caption)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.positional_encoding.encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for decoder_block in self.decoder_blocks:
            x, _, _ = decoder_block(x, encoder_output, padding_mask, look_ahead_mask, training)
        return x


class TransformerGenerator(tf.keras.Model):
    def __init__(self, token_vocab_size, style_vocab_size, model_dim, style_dim, pffn_dim, z_dim, encoder_blocks,
                 decoder_blocks, num_attention_heads, max_pe, dropout, stylize):
        super().__init__()
        self.token_vocab_size = token_vocab_size
        self.encoder = Encoder(blocks=encoder_blocks, model_dim=model_dim, style_dim=style_dim, pffn_dim=pffn_dim,
                               num_heads=num_attention_heads, style_vocab_size=style_vocab_size, dropout=dropout,
                               stylize=stylize)
        self.decoder = Decoder(blocks=decoder_blocks, model_dim=model_dim, pffn_dim=pffn_dim,
                               num_heads=num_attention_heads, token_vocab_size=token_vocab_size,
                               dropout=dropout, max_pe=max_pe)
        self.final_layer = tf.keras.layers.Dense(token_vocab_size, activation="linear")
        self.forward(tf.ones((3, *InceptionResNetEncoder.IMAGE_FEATURE_SHAPE)), tf.ones((3, 10)), tf.ones((3,)), True)

    def call(self, image_feature, caption, style, training, encoder_padding_mask, decoder_padding_mask,
             look_ahead_mask):
        encoder_output = self.encoder(image_feature, style, encoder_padding_mask, training)
        decoder_output = self.decoder(encoder_output, caption, decoder_padding_mask, look_ahead_mask, training)
        logits = self.final_layer(decoder_output)
        return logits

    def forward(self, image_feature, caption, style, training):
        image_feature = self._reshape_image_feature(image_feature)
        encoder_padding_mask, decoder_padding_mask, look_ahead_mask = self._create_masks(caption)
        return self.call(image_feature, caption, style, training, encoder_padding_mask, decoder_padding_mask,
                         look_ahead_mask)

    def sample(self, image_feature, initial_sequence, style, sequence_length, mode, n_samples, training, sos, eos):
        if mode not in ["stochastic", "deterministic"]:
            raise ValueError(f"Mode must be one of - stochastic, deterministic")

        image_feature = self._reshape_image_feature(image_feature)
        initial_sequence_length = initial_sequence.shape[1]
        samples = []
        sample_logits = []
        for n in range(n_samples):
            sequence = initial_sequence
            logits = None
            for t in range(initial_sequence_length, sequence_length):
                encoder_padding_mask, decoder_padding_mask, look_ahead_mask = self._create_masks(sequence)
                logits = self.call(image_feature, sequence, style, training, encoder_padding_mask, decoder_padding_mask,
                                   look_ahead_mask)
                logits_t = logits[:, -1:, :]
                if mode == "deterministic":
                    token = tf.cast(tf.argmax(logits_t, axis=-1), tf.int64)
                else:
                    token = tfp.distributions.Categorical(logits=logits_t, dtype=tf.int64).sample()
                sequence = tf.concat([sequence, token], axis=1)

            oh = tf.expand_dims(tf.math.log(
                tf.one_hot([sos] * logits.shape[0], depth=logits.shape[-1], dtype=tf.float32)
            ), axis=1)
            logits = tf.concat([oh, logits], axis=1)

            eos_mask = self._get_eos_mask(sequence, eos)
            samples.append(sequence * tf.cast(eos_mask, dtype=tf.int64))
            sample_logits.append(logits * tf.cast(
                tf.repeat(tf.expand_dims(eos_mask, axis=2), self.token_vocab_size, axis=2),
                dtype=tf.float32
            ))

        return samples, sample_logits

    def beam_search(self, image_feature, style, sequence_length, beam_size, sos, eos):
        batch_size = image_feature.shape[0]

        image_feature = self._reshape_image_feature(image_feature)
        image_feature = tf.reshape(tf.tile(image_feature, [1, beam_size, 1]),
                                   ((batch_size * beam_size), *image_feature.shape[1:]))
        style = tf.tile(style, [beam_size])

        sequences = tf.constant(sos, shape=((batch_size * beam_size), 1), dtype=tf.int64)  # (8*5, 1)
        sequences_logits = tf.constant(0, shape=((batch_size * beam_size), 1), dtype=tf.float32)  # (8*5, 1)

        for t in range(sequence_length - 1):
            current_tokens = sequences[:, -1]
            current_logits = sequences_logits[:, -1]

            encoder_padding_mask, decoder_padding_mask, look_ahead_mask = self._create_masks(sequences)
            logits = self.call(image_feature, sequences, style, False, encoder_padding_mask, decoder_padding_mask,
                               look_ahead_mask)
            logits_t = logits[:, -1, :]

            logits_t = tf.math.log(tf.nn.softmax(logits_t))
            logits_l1, indices_l1 = tf.math.top_k(logits_t, k=beam_size)  # (8*5, 5)
            logits_l1 += tf.transpose(tf.reshape(
                tf.tile(current_logits, [beam_size]), (beam_size, batch_size * beam_size)
            ))
            logits_l1 = tf.reshape(logits_l1, (batch_size, beam_size * beam_size))  # (8, 5*5)
            indices_l1 = tf.cast(indices_l1, dtype=tf.int64)
            indices_l1 = tf.reshape(indices_l1, (batch_size, beam_size * beam_size))  # (8, 5*5)
            logits_l2, indices_l2 = tf.math.top_k(logits_l1, k=beam_size)  # (8, 5)

            next_tokens = tf.gather_nd(indices_l1, tf.stack([
                tf.repeat(tf.range(batch_size), beam_size, axis=0),
                tf.reshape(indices_l2, (-1,))
            ], axis=1))

            current_reordered_indices = tf.math.reduce_sum(tf.stack([
                tf.repeat(tf.range(batch_size), beam_size, axis=0) * beam_size,
                tf.reshape(tf.cast(indices_l2 / 5, dtype=tf.int32), (-1,))
            ], axis=1), axis=1)
            current_tokens = tf.gather(current_tokens, current_reordered_indices)

            sequences = tf.slice(sequences, [0, 0], [batch_size * beam_size, sequences.shape[1] - 1])
            sequences = tf.concat([
                sequences, tf.expand_dims(current_tokens, axis=1), tf.expand_dims(next_tokens, axis=1)
            ], axis=1)
            sequences_logits = tf.concat([
                sequences_logits, tf.reshape(logits_l2, (-1, 1))
            ], axis=1)

        eos_mask = self._get_eos_mask(sequences, eos)
        sequences = sequences * tf.cast(eos_mask, dtype=tf.int64)
        sequences = tf.reshape(sequences, (batch_size, beam_size, -1))
        sequences_logits = tf.reshape(sequences_logits[:, -1], (batch_size, beam_size))
        return sequences, sequences_logits

    @staticmethod
    def _reshape_image_feature(image_feature):
        return tf.reshape(image_feature, shape=(image_feature.shape[0], -1, image_feature.shape[3]))

    @staticmethod
    def _create_masks(caption):
        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask

        def create_padding_mask(seq):
            seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
            return seq[:, tf.newaxis, tf.newaxis, :]

        look_ahead_mask = create_look_ahead_mask(caption.shape[1])
        caption_padding_mask = create_padding_mask(caption)
        look_ahead_mask = tf.maximum(caption_padding_mask, look_ahead_mask)
        return None, None, look_ahead_mask

    @staticmethod
    def _get_eos_mask(sequence, eos):
        batch_size, sequence_length = sequence.shape[0], sequence.shape[1]
        mask = tf.tensor_scatter_nd_update(
            sequence, tf.stack([tf.range(batch_size), tf.constant(sequence_length - 1, shape=(batch_size,))], axis=1),
            tf.constant(eos, dtype=tf.int64, shape=(batch_size,))
        )
        mask = tf.broadcast_to(tf.expand_dims(tf.argmax(tf.cast(mask == eos, tf.int64), axis=1), axis=1),
                               (batch_size, sequence_length))
        mask = mask >= tf.broadcast_to(tf.range(sequence_length, dtype=tf.int64), (batch_size, sequence_length))
        return mask
