"""
predict.py
----------
Inference script — takes a mid-innings match situation as input
and outputs the predicted final total.

Usage:
    python src/predict.py \
        --venue "M Chinnaswamy Stadium" \
        --bat_team "Chennai Super Kings" \
        --bowl_team "Royal Challengers Bangalore" \
        --batsman "MS Dhoni" \
        --bowler "Yuzvendra Chahal" \
        --runs 78 \
        --wickets 2 \
        --overs 10.0 \
        --striker 1
"""

import argparse
import numpy as np

from preprocess import load_artifacts
from model import load_model


def encode_input(user_input: dict, encoders: dict) -> list:
    """
    Encode a single match-state dict into the feature vector
    expected by the model.

    Feature order must match training: bat_team, bowl_team, venue,
    runs, wickets, overs, striker, batsman, bowler
    """
    cat_fields = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']

    for field in cat_fields:
        val = user_input[field]
        le  = encoders[field]
        if val not in le.classes_:
            raise ValueError(
                f"Unknown value '{val}' for field '{field}'.\n"
                f"Valid options: {list(le.classes_)}"
            )
        user_input[field] = le.transform([val])[0]

    # Build feature vector in exact training order
    feature_vector = [
        user_input['bat_team'],
        user_input['bowl_team'],
        user_input['venue'],
        user_input['runs'],
        user_input['wickets'],
        user_input['overs'],
        user_input['striker'],
        user_input['batsman'],
        user_input['bowler'],
    ]
    return feature_vector


def predict(user_input: dict, model_dir='models/') -> int:
    """
    Given a match situation dict, return the predicted total score.
    """
    encoders, scaler = load_artifacts(model_dir)
    model = load_model(f'{model_dir}/ipl_model.h5')

    feature_vector = encode_input(user_input, encoders)
    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)

    predicted = model.predict(X_scaled, verbose=0)
    return int(predicted[0][0])


def main():
    parser = argparse.ArgumentParser(description='IPL Score Predictor')
    parser.add_argument('--venue',     required=True)
    parser.add_argument('--bat_team',  required=True)
    parser.add_argument('--bowl_team', required=True)
    parser.add_argument('--batsman',   required=True)
    parser.add_argument('--bowler',    required=True)
    parser.add_argument('--runs',      type=int,   required=True)
    parser.add_argument('--wickets',   type=int,   required=True)
    parser.add_argument('--overs',     type=float, required=True)
    parser.add_argument('--striker',   type=int,   default=1, choices=[0, 1])
    args = parser.parse_args()

    user_input = vars(args)

    print("\n--- Match Situation ---")
    for k, v in user_input.items():
        print(f"  {k:12}: {v}")
    print()

    score = predict(user_input)
    print(f"  -> Predicted Total: {score} runs\n")


if __name__ == '__main__':
    main()
