# 🏏 IPL Score Prediction using Deep Learning

A deep learning-based regression model that predicts the **final innings total** of an IPL batting team in real time, given the current match situation.

---

## 📌 Project Overview

The Indian Premier League (IPL) is one of the most-watched cricket leagues in the world. Predicting the score during a live innings is valuable for broadcasters, fantasy leagues, and strategic team planning.

Traditional run-rate extrapolation ignores context like:
- Which batsman is at the crease
- Who is bowling
- How many wickets have fallen
- The historical performance of teams at a particular venue

This project uses a **multi-layer feedforward neural network** (TensorFlow + Keras) trained on IPL ball-by-ball data for the 2026 season to capture these patterns and make accurate predictions.

---

## 📊 Dataset

- **Source:** [IPL Dataset on Kaggle](https://www.kaggle.com/datasets/nowke9/ipldata)
- **Season:** 2026 (Mock Data generated via script)
- **Format:** Ball-by-ball records (one row per ball bowled)
- **Size:** 5,000 rows

### Key Features Used

| Feature | Description |
|---|---|
| `venue` | Ground where match is played |
| `bat_team` | Batting team name |
| `bowl_team` | Bowling team name |
| `batsman` | Striker batsman name |
| `bowler` | Bowler name |
| `runs` | Runs scored so far in innings |
| `wickets` | Wickets fallen so far |
| `overs` | Overs completed |
| `striker` | Strike indicator (0 or 1) |
| `total` | **Target — final innings total** |

> **Note:** Place the downloaded CSV as `data/ipl_dataset.csv` before running the notebooks.

---

## 🗂️ Project Structure

```
ipl-score-prediction/
│
├── data/
│   └── ipl_dataset.csv          # Raw dataset (download from Kaggle)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   # Encoding, feature selection, scaling
│   └── 03_model_training.ipynb  # Model building, training, evaluation
│
├── src/
│   ├── preprocess.py            # Preprocessing utilities
│   ├── model.py                 # Model definition
│   ├── train.py                 # Training script
│   └── predict.py               # Inference / prediction script
│
├── models/
│   └── ipl_model.h5             # Saved trained model (generated after training)
│
├── outputs/
│   └── training_loss.png        # Loss curve plot (generated after training)
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

```
Input (9 features)
       │
  Dense(512, ReLU)
       │
  Dense(216, ReLU)
       │
   Dense(1, Linear)
       │
  Predicted Total
```

| Parameter | Value |
|---|---|
| Loss Function | Huber Loss (δ = 1.0) |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 64 |
| Train / Test Split | 70% / 30% |

**Why Huber Loss?**  
T20 innings can have explosive finishes (200+ totals). Huber loss is robust to these outliers — it behaves like MSE for small errors and like MAE for large errors.

---

## 📈 Results

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ~14.4 runs |
| Evaluation Set | 30% held-out test set |

An MAE of ~14 runs means the model is off by roughly one over's worth of runs on average — reasonable for live T20 prediction.

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/tech.indro/ipl-score-prediction.git
cd ipl-score-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download `ipl_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/nowke9/ipldata) and place it inside the `data/` folder.

### 4. Train the model
```bash
python src/train.py
```

### 5. Make a prediction
```bash
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
```

### 6. (Optional) Run Jupyter notebooks
```bash
jupyter notebook notebooks/
```

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow 2.x / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📉 Limitations

- Currently uses a mock dataset for the 2026 season.
- No weather, toss result, or pitch condition data is included.
- Point estimate only — no prediction confidence interval.
- Player performance is assumed stationary across seasons.

---

## 🔮 Future Improvements

- Replace mock data with a real dataset for the 2026 season once available
- Add LSTM-based model for ball-by-ball sequence modelling
- Include toss, weather, and team form features
- Deploy as a REST API using FastAPI or Flask
- Build a Streamlit web app for interactive predictions

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---