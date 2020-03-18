import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.resnet = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights="imagenet")
        # TODO: average pooling to resize to 14 x 14?
        # TODO: fine tune last few convolution blocks?
        for layer in self.resnet.layers:
            layer.trainable = False

    def call(self, inp):
        x = tf.keras.applications.resnet_v2.preprocess_input(inp)
        x = self.resnet(x)
        return x

    @property
    def output_shape(self):
        return self.resnet.layers[-1].output_shape


class EncoderGeneratorAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.encoder_attention = tf.keras.layers.Dense(units)
        self.generator_attention = tf.keras.layers.Dense(units)
        self.relu = tf.keras.layers.ReLU()
        self.full_attention = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, encoder_output, carry_state):
        attention_1 = self.encoder_attention(encoder_output)
        attention_2 = self.generator_attention(carry_state)
        attention_alpha = self.relu(attention_1 + tf.expand_dims(attention_2, axis=1))
        attention_alpha = self.full_attention(attention_alpha)
        attention_alpha = tf.squeeze(attention_alpha, axis=2)
        attention_alpha = self.softmax(attention_alpha)
        attention_weighted_encoding = tf.reduce_sum((encoder_output * tf.expand_dims(attention_alpha, axis=2)), axis=1)
        return attention_weighted_encoding, attention_alpha


class Generator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, attention_units, lstm_units, encoder_units, lstm_dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_units = embedding_units
        self.attention_units = attention_units
        self.lstm_units = lstm_units
        self.encoder_units = encoder_units
        self.lstm_dropout = lstm_dropout

        self.attention = EncoderGeneratorAttention(units=self.attention_units)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_units)
        self.lstm = tf.keras.layers.LSTMCell(units=lstm_units, activation="tanh")
        self.dense_init_carry_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_init_memory_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_f_beta = tf.keras.layers.Dense(units=encoder_units, activation="sigmoid")
        self.dense_lstm_output = tf.keras.layers.Dense(units=self.vocab_size, activation="softmax")
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        self.forward(tf.ones((1, 10, 10, 2048)), tf.ones((1, 10)))  # initialize weights

    def call(self, encoder_output, sequences_t, memory_state, carry_state, training=False):
        embeddings = self.embedding(sequences_t)
        attention_weighted_encoding, attention_alpha = self.attention(encoder_output, carry_state)
        beta_gate = self.dense_f_beta(carry_state)
        attention_weighted_encoding *= beta_gate
        lstm_inputs = tf.concat([embeddings, attention_weighted_encoding], axis=1)
        _, (memory_state, carry_state) = self.lstm(lstm_inputs, [memory_state, carry_state])
        prediction = self.dense_lstm_output(self.dropout(carry_state, training=training))
        return prediction, attention_alpha, memory_state, carry_state

    def forward(self, encoder_output, sequences, teacher_forcing_rate=1, training=False):
        predictions, attention_alphas = [], []
        encoder_output = self._reshape_encoder_output(encoder_output)
        memory_state, carry_state = self.init_lstm_states(encoder_output)
        for t in range(sequences.shape[-1]):
            sequences_t = sequences[:, t]
            if t > 0 and np.random.uniform() > teacher_forcing_rate:
                sequences_t = tf.argmax(predictions[:, t - 1, :], axis=1)
            prediction, attention_alpha, memory_state, carry_state = self.call(encoder_output, sequences_t,
                                                                               memory_state, carry_state,
                                                                               training=training)
            predictions.append(prediction)
            attention_alphas.append(attention_alpha)
        return tf.stack(predictions, axis=1), tf.stack(attention_alphas, axis=1)

    def generate_caption(self, encoder_output, mode, start_id, end_id, max_len=20):
        # TODO: Implement beam_search
        if mode not in ["stochastic", "deterministic", "beam_search", "mcts"]:
            raise ValueError(f"Caption generation mode {mode} is not valid")
        batch_size = encoder_output.shape[0]
        sequences = [tf.ones(batch_size, dtype=tf.int64) * start_id]
        encoder_output = self._reshape_encoder_output(encoder_output)
        memory_state, carry_state = self.init_lstm_states(encoder_output)
        keep_generating_mask = tf.ones(batch_size, dtype=tf.int64)
        for t in range(1, max_len):
            prediction, _, memory_state, carry_state = self.call(encoder_output, sequences[t - 1],
                                                                 memory_state, carry_state)
            if mode == "stochastic":
                dist = tfp.distributions.Categorical(probs=prediction)
                tokens = dist.sample()
            elif mode == "deterministic":
                tokens = tf.argmax(prediction, axis=1)
            tokens *= keep_generating_mask
            sequences.append(tokens)
            keep_generating_mask = tf.cast(tokens != end_id, dtype=tf.int64) * keep_generating_mask
        return tf.stack(sequences, axis=1)

    def sample(self, encoder_output, initial_values, sequence_length, n_samples):
        encoder_output = self._reshape_encoder_output(encoder_output)
        init_sequence_length = initial_values.shape[1]
        init_sequences = [tf.argmax(initial_values[:, 0, :], axis=1)]
        init_memory_state, init_carry_state = self.init_lstm_states(encoder_output)
        for t in range(1, init_sequence_length):
            prediction, _, init_memory_state, init_carry_state = self.call(encoder_output, init_sequences[t - 1],
                                                                           init_memory_state, init_carry_state)
            init_sequences.append(tf.argmax(prediction, axis=1))

        samples = []
        for n in range(n_samples):
            sequences = [tf.identity(s) for s in init_sequences]
            memory_state, carry_state = tf.identity(init_memory_state), tf.identity(init_carry_state)
            for t in range(init_sequence_length, sequence_length):
                prediction, _, memory_state, carry_state = self.call(encoder_output, sequences[t - 1],
                                                                     memory_state, carry_state)
                sequences.append(tfp.distributions.Categorical(probs=prediction, dtype=tf.int64).sample())
            samples.append(tf.stack(sequences, axis=1))
        return samples

    def init_lstm_states(self, encoder_output):
        # TODO: add random vector z here?
        mean_encoder_output = tf.reduce_mean(encoder_output, axis=1)
        memory_state = self.dense_init_memory_state(mean_encoder_output)
        carry_state = self.dense_init_carry_state(mean_encoder_output)
        return memory_state, carry_state

    @staticmethod
    def _reshape_encoder_output(encoder_output):
        return tf.reshape(encoder_output, shape=(encoder_output.shape[0], -1, encoder_output.shape[3]))


class Discriminator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, lstm_units):
        super().__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_units)
        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_units, activation="tanh", return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, activation="tanh"))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.call(tf.ones((1, 10, 10, 2048)), tf.ones((1, 10)))  # initialize weights

    def call(self, encoder_output, sequences, training=False):
        encoder_output = self.pooling(encoder_output)
        embeddings = self.embedding(sequences)
        lstm1_out = self.lstm1(embeddings)
        lstm2_out = self.lstm2(lstm1_out)
        all_features = tf.concat([encoder_output, lstm2_out], axis=1)
        x = self.dense1(all_features)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x
