"""TensorFlow models for binary review classification."""
import tensorflow as tf

L = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="IMDb")
class AttentionPooling(L.Layer):
    """Learn a scalar relevance score per token; exclude padding from pooling."""

    def __init__(self, width=32, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.width = width
        self.hidden = L.Dense(width, activation="tanh")
        self.score = L.Dense(1, use_bias=False)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        states, token_ids = inputs
        logits = tf.cast(self.score(self.hidden(states)), tf.float32)
        keep = token_ids[..., None] != 0
        logits = tf.where(keep, logits, -1e9)
        weights = tf.nn.softmax(logits, axis=1) * tf.cast(keep, tf.float32)
        weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=1, keepdims=True))
        return tf.reduce_sum(states * tf.cast(weights, states.dtype), axis=1)

    def get_config(self):
        return {**super().get_config(), "width": self.width}


def build_model(architecture, vocab_size, length, dropout):
    """Build one recurrent encoder with a common attention and classifier head."""
    cells = {"GRU": L.GRU, "LSTM": L.LSTM}
    ids = tf.keras.Input((length,), dtype="int32", name="token_ids")
    x = L.Embedding(vocab_size, 64, mask_zero=True, name="embedding")(ids)
    states = L.Bidirectional(cells[architecture](48, return_sequences=True),
                             name="bidirectional_encoder")(x)
    x = AttentionPooling(32, name="attention_pool")([states, ids])
    x = L.Dense(32, activation="relu", name="review_features")(x)
    x = L.Dropout(dropout, name="classifier_dropout")(x)
    p = L.Dense(1, activation="sigmoid", name="positive_probability")(x)
    model = tf.keras.Model(ids, p, name=f"bi{architecture.lower()}_attention")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return model
