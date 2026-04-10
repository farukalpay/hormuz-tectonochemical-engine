from __future__ import annotations

from .config import AppConfig


def build_forecaster(config: AppConfig, n_features: int, n_targets: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(config.data.lookback_steps, n_features), name="driver_window")
    x = tf.keras.layers.LayerNormalization(name="window_norm")(inputs)
    x = tf.keras.layers.LSTM(
        config.training.lstm_units[0],
        return_sequences=True,
        dropout=config.training.dropout,
        name="lstm_encoder_1",
    )(x)
    x = tf.keras.layers.LSTM(
        config.training.lstm_units[1],
        return_sequences=True,
        dropout=config.training.dropout,
        name="lstm_encoder_2",
    )(x)
    x = tf.keras.layers.LSTM(
        config.training.lstm_units[2],
        return_sequences=True,
        dropout=config.training.dropout,
        name="lstm_encoder_3",
    )(x)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=config.training.attention_heads,
        key_dim=max(8, config.training.lstm_units[2] // config.training.attention_heads),
        dropout=config.training.dropout,
        name="temporal_attention",
    )(x, x)
    x = tf.keras.layers.Add(name="attention_residual")([x, attn])
    x = tf.keras.layers.LayerNormalization(name="attention_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="context_pool")(x)
    x = tf.keras.layers.Dense(96, activation="swish", name="dense_projection")(x)
    x = tf.keras.layers.Dropout(config.training.dropout, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(n_targets, name="observable_forecast")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hte_lstm_attention_forecaster")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
