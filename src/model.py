"""
model.py
--------
Defines and compiles the deep learning model for IPL score prediction.
"""

import tensorflow as tf
import keras


def build_model(input_dim: int) -> keras.Model:
    """
    Builds a 3-layer feedforward neural network for regression.

    Architecture:
        Input  →  Dense(512, ReLU)
               →  Dense(216, ReLU)
               →  Dense(1, Linear)   ← predicted total

    Loss: Huber (robust to outlier scores like 220+ totals)
    Optimizer: Adam (adaptive learning rate)

    Args:
        input_dim: Number of input features (9 in this project)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(216, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])

    # Huber loss is more stable than MSE when there are high-scoring outlier innings
    huber = tf.keras.losses.Huber(delta=1.0)

    model.compile(optimizer='adam', loss=huber)

    model.summary()
    return model


def load_model(model_path='models/ipl_model.h5') -> keras.Model:
    """Load a previously saved model from disk."""
    model = keras.models.load_model(
        model_path,
        custom_objects={'huber_loss': tf.keras.losses.Huber(delta=1.0)}
    )
    print(f"[INFO] Model loaded from '{model_path}'")
    return model
