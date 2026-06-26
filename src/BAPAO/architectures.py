from __future__ import annotations

from collections.abc import Sequence

import keras
import tensorflow as tf

# very compute expensive, but does allow CVD models to train without NaN loss values.
# They have very large ranges thus require the extra precision to avoid underflow/overflow
# issues during training.
# TODO: Normalise the input parameters to avoid this requirement and speed up training.
tf.keras.mixed_precision.set_global_policy("float32")

# the way faster option, but doesnt work for CVD models due to NaN loss values during training.
# tf.keras.mixed_precision.set_global_policy("mixed_float16")


def build_parameterized_dnn(
    model_columns: Sequence[str],
    width: int = 128,
    depth: int = 5,
):
    class ParameterizedDNN(keras.Model):
        def __init__(self):
            super().__init__()
            self.model_columns = tuple(model_columns)
            self.width = width
            self.depth = depth

            # self.input_scaler = keras.layers.Normalization(axis=-1, name="input_scaler")

            self.hidden = [
                keras.layers.Dense(
                    width,
                    activation="gelu",
                    kernel_initializer="he_normal",
                    name=f"hidden_{idx}",
                )
                for idx in range(depth)
            ]

            #  Force float32 on the output layer for mixed_precision stability
            self.out = keras.layers.Dense(
                1,
                name="out",
                kernel_initializer="zeros",
                bias_initializer=keras.initializers.Constant(0.5),  # type: ignore
                dtype="float32",  # float32 for mixed_precision
            )

        def call(self, inputs):
            xi = inputs[:, 0:1]
            k_log = inputs[:, 1:2]

            features = [
                (xi - 0.5) * 2.0,
                (k_log + 2.5) / 2.5,
            ]

            if inputs.shape[1] > 2:
                extra_params = inputs[:, 2:]
                safe_params = tf.maximum(extra_params, 1e-10)
                extra_params_n = (tf.math.log(safe_params) + 11.5) / 5.0
                features.append(extra_params_n)

            x = tf.concat(features, axis=1)
            # x = self.input_scaler(x)

            for layer in self.hidden:
                x = layer(x)
            return self.out(x)

    model = ParameterizedDNN()
    _ = model(tf.zeros((1, 1 + len(tuple(model_columns))), dtype=tf.float32))
    return model
