# 🎾 Tennis Betting — Match Analysis and Prediction Engine

A project designed to analyze tennis match data and build predictive models to optimize betting strategies.

---

## 📦 What's Included

- `data_exploration.ipynb` – Initial data exploration and visual insights
- `data_processing.py` – Data cleaning, formatting, and feature engineering
- `models.py` – Model training and evaluation (logistic regression, etc.)
- `test.py` & `test.ipynb` – Prediction testing and accuracy evaluation
- `pickles/` – Stored models and data (for reuse)
- `Scrapper/` – Match data scraping utilities

---

## 🔍 Goals

- Analyze historical tennis data (players, matches, stats)
- Engineer meaningful features (e.g. player form, head-to-head, surface)
- Train models to predict match outcomes
- Test betting scenarios using model predictions

---

## ⚙️ Tech Stack

- **Language**: Python
- **Tools**: Pandas, Scikit-learn, Jupyter
- **Storage**: Pickle serialization for fast reuse

---

## 🚀 Getting Started

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the exploration notebook

```bash
jupyter notebook data_exploration.ipynb
```

### 3. Train and test models

```bash
python models.py
python test.py
```

---

## 🧠 Future Ideas

- Improve scraping for real-time data
- Add probabilistic betting simulation
- Include bankroll management strategies

---

## 📄 License

This project is under the MIT License. See [LICENSE](./LICENSE).

---

## 👤 Author

Developed by [Lorys Hamadache](https://github.com/LorysHamadache)
