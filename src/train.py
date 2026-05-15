"""
train.py
--------
End-to-end training script for the IPL Score Prediction model.

Usage:
    python src/train.py
    python src/train.py --data data/ipl_dataset.csv --epochs 15 --batch_size 64
"""

import os
import argparse
import matplotlib.pyplot as plt

from preprocess import run_pipeline
from model import build_model


def plot_loss(history, out_path='outputs/training_loss.png'):
    """Save a training vs validation loss plot."""
    os.makedirs('outputs', exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history.history['loss'],     label='Training Loss',   color='#0d2137', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#ffa500', linewidth=2, linestyle='--')
    plt.title('Model Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Loss plot saved to '{out_path}'")


def train(data_path, epochs, batch_size):
    print("\n========== IPL Score Prediction — Training ==========\n")

    # Step 1: Preprocess
    X_train, X_test, y_train, y_test, encoders, scaler = run_pipeline(data_path)

    # Step 2: Build model
    model = build_model(input_dim=X_train.shape[1])

    # Step 3: Train
    print("\n[INFO] Starting training...\n")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Step 4: Evaluate
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    predictions = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, predictions)
    print(f"\n[RESULT] Mean Absolute Error on test set: {mae:.4f} runs")

    # Step 5: Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/ipl_model.h5')
    print("[INFO] Model saved to 'models/ipl_model.h5'")

    # Step 6: Plot loss
    plot_loss(history)

    print("\n========== Training Complete ==========\n")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IPL Score Prediction Model')
    parser.add_argument('--data',       type=str, default='data/ipl_dataset.csv')
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size)
