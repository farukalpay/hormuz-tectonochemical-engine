from __future__ import annotations

import numpy as np

from .config import AppConfig


def _baseline_target_projection(
    config: AppConfig,
    inputs,
    *,
    n_features: int,
    n_targets: int,
    target_feature_indices: tuple[int, ...],
    target_feature_mean: np.ndarray,
    target_feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
):
    import tensorflow as tf

    baseline_kernel = np.zeros((n_features, n_targets), dtype=np.float32)
    baseline_bias = np.zeros((n_targets,), dtype=np.float32)
    for target_index, feature_index in enumerate(target_feature_indices):
        baseline_kernel[feature_index, target_index] = target_feature_std[target_index] / target_std[target_index]
        baseline_bias[target_index] = (target_feature_mean[target_index] - target_mean[target_index]) / target_std[target_index]

    last_row = tf.keras.layers.Cropping1D(
        cropping=(config.data.lookback_steps - 1, 0),
        name="latest_window_row",
    )(inputs)
    last_row = tf.keras.layers.Flatten(name="latest_window_row_flat")(last_row)
    return tf.keras.layers.Dense(
        n_targets,
        activation=None,
        trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(baseline_kernel),
        bias_initializer=tf.keras.initializers.Constant(baseline_bias),
        name="baseline_target_projection",
    )(last_row)


def _compile_forecaster(model, config: AppConfig):
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training.learning_rate,
        epsilon=config.training.optimizer_epsilon,
        clipnorm=config.training.optimizer_clipnorm,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        run_eagerly=config.training.run_eagerly,
        jit_compile=config.training.jit_compile,
        steps_per_execution=config.training.steps_per_execution,
    )
    return model


def _build_stacked_attention_forecaster(
    config: AppConfig,
    n_features: int,
    n_targets: int,
    *,
    target_feature_indices: tuple[int, ...],
    target_feature_mean: np.ndarray,
    target_feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(config.data.lookback_steps, n_features), name="driver_window")
    x = tf.keras.layers.LayerNormalization(name="window_norm")(inputs)
    x = tf.keras.layers.LSTM(
        config.training.lstm_units[0],
        return_sequences=True,
        dropout=config.training.dropout,
        unroll=config.training.lstm_unroll,
        use_cudnn=config.training.lstm_use_cudnn,
        name="lstm_encoder_1",
    )(x)
    x = tf.keras.layers.LSTM(
        config.training.lstm_units[1],
        return_sequences=True,
        dropout=config.training.dropout,
        unroll=config.training.lstm_unroll,
        use_cudnn=config.training.lstm_use_cudnn,
        name="lstm_encoder_2",
    )(x)
    x = tf.keras.layers.LSTM(
        config.training.lstm_units[2],
        return_sequences=True,
        dropout=config.training.dropout,
        unroll=config.training.lstm_unroll,
        use_cudnn=config.training.lstm_use_cudnn,
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
    residual = tf.keras.layers.Dense(n_targets, activation="tanh", name="observable_residual")(x)

    baseline = _baseline_target_projection(
        config,
        inputs,
        n_features=n_features,
        n_targets=n_targets,
        target_feature_indices=target_feature_indices,
        target_feature_mean=target_feature_mean,
        target_feature_std=target_feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    residual = tf.keras.layers.Rescaling(
        scale=config.training.residual_output_scale,
        name="residual_output_scale",
    )(residual)
    outputs = tf.keras.layers.Add(name="observable_forecast")([baseline, residual])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hte_lstm_attention_forecaster")
    return _compile_forecaster(model, config)


def _build_safe_recurrent_forecaster(
    config: AppConfig,
    n_features: int,
    n_targets: int,
    *,
    target_feature_indices: tuple[int, ...],
    target_feature_mean: np.ndarray,
    target_feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
):
    import tensorflow as tf

    safe_units = max(8, int(config.training.lstm_units[-1]))
    dense_units = max(32, safe_units)

    inputs = tf.keras.Input(shape=(config.data.lookback_steps, n_features), name="driver_window")
    x = tf.keras.layers.LSTM(
        safe_units,
        return_sequences=False,
        dropout=config.training.dropout,
        unroll=config.training.lstm_unroll,
        use_cudnn=config.training.lstm_use_cudnn,
        name="lstm_safe_encoder",
    )(inputs)
    x = tf.keras.layers.Dense(dense_units, activation="tanh", name="safe_dense_projection")(x)
    residual = tf.keras.layers.Dense(n_targets, activation="tanh", name="safe_observable_residual")(x)

    baseline = _baseline_target_projection(
        config,
        inputs,
        n_features=n_features,
        n_targets=n_targets,
        target_feature_indices=target_feature_indices,
        target_feature_mean=target_feature_mean,
        target_feature_std=target_feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    residual = tf.keras.layers.Rescaling(
        scale=config.training.residual_output_scale,
        name="safe_residual_output_scale",
    )(residual)
    outputs = tf.keras.layers.Add(name="safe_observable_forecast")([baseline, residual])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hte_safe_recurrent_forecaster")
    return _compile_forecaster(model, config)


def build_forecaster(
    config: AppConfig,
    n_features: int,
    n_targets: int,
    *,
    target_feature_indices: tuple[int, ...],
    target_feature_mean: np.ndarray,
    target_feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
):
    variant = config.training.model_variant.strip().lower()
    if variant in {"stacked_attention", "default"}:
        return _build_stacked_attention_forecaster(
            config,
            n_features,
            n_targets,
            target_feature_indices=target_feature_indices,
            target_feature_mean=target_feature_mean,
            target_feature_std=target_feature_std,
            target_mean=target_mean,
            target_std=target_std,
        )
    if variant in {"safe_recurrent", "safe-recurrent"}:
        return _build_safe_recurrent_forecaster(
            config,
            n_features,
            n_targets,
            target_feature_indices=target_feature_indices,
            target_feature_mean=target_feature_mean,
            target_feature_std=target_feature_std,
            target_mean=target_mean,
            target_std=target_std,
        )
    raise ValueError(
        f"training.model_variant must be one of: stacked_attention, safe_recurrent (got {config.training.model_variant!r})"
    )
