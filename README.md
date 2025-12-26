# 🌍 Calamity News Analysis Project

## 📌 Overview
This is a  web/text mining data science project focused on collecting, processing, and analyzing global news related to natural calamities such as floods, earthquakes, droughts, hurricanes, wildfires, and other disaster events.

The project applies Natural Language Processing (NLP) and machine learning techniques to classify calamity types and generate country-level insights along with visual analytics.

This repository is maintained for personal learning, experimentation, and portfolio reference.

---

## 📂 Project Structure

```
calamity-news-analysis/
│
├── README.md
├── scrape_news.py              # Main pipeline entry point
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning and preprocessing
│   ├── ner_countries.py        # Country extraction using NER
│   ├── label_calamity.py       # Calamity labeling logic
│   ├── features_models.py      # Feature engineering and ML models
│   └── aggregate_visuals.py    # Aggregation and visualization logic
│
├── sources/
│   ├── gdelt_fetch.py          # GDELT news data collection
│   ├── google_rss_fetch.py     # Google RSS news scraping
│   └── newsapi_fetch.py        # NewsAPI data fetching
│
└── results/
    ├── figures/                # Generated plots and visualizations
    ├── tables/                 # Aggregated CSV outputs
    ├── confusion_matrix_lr.png # Logistic Regression results
    ├── confusion_matrix_nb.png # Naive Bayes results
    ├── model_report.txt        # Model evaluation report
    └── test_predictions.csv    # Model predictions

```


---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Natural Language Processing (NLP)
- News APIs and RSS feeds

---

## ⚙️ Project Functionality
- Collects news articles related to natural disasters
- Cleans and preprocesses text data
- Extracts country information using Named Entity Recognition (NER)
- Classifies articles by calamity type
- Aggregates results at country and event level
- Generates visualizations and summary tables
- Evaluates models using confusion matrices and classification reports

---

## 📊 Outputs
- Heatmaps showing calamity distribution by country
- Top-10 calamity visualizations by type
- Confusion matrices for model evaluation
- CSV tables with aggregated results and predictions

---

## 🚀 Usage
This project is intended for personal use and exploration.

The main pipeline can be executed using:
```bash
python scrape_news.py
