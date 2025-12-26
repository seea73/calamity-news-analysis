# 🌍 Calamity News Analysis Project

## 📌 Overview
This is a  web/text mining data science project focused on collecting, processing, and analyzing global news related to natural calamities such as floods, earthquakes, droughts, hurricanes, wildfires, and other disaster events.

The project applies Natural Language Processing (NLP) and machine learning techniques to classify calamity types and generate country-level insights along with visual analytics.

This repository is maintained for personal learning, experimentation, and portfolio reference.

---

## 📂 Project Structure

FINAL_PROJECT/
├── src/
│   ├── preprocess.py
│   ├── ner_countries.py
│   ├── label_calamity.py
│   ├── features_models.py
│   └── aggregate_visuals.py
│
├── sources/
│   ├── gdelt_fetch.py
│   ├── google_rss_fetch.py
│   └── newsapi_fetch.py
│
├── results/
│   ├── figures/
│   │   ├── heatmap_calamity_country.png
│   │   ├── top_10_calamities.png
│   │   └── confusion_matrix.png
│   │
│   ├── tables/
│   │   ├── aggregated_calamities.csv
│   │   └── country_calamity_counts.csv
│   │
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_nb.png
│   ├── model_report.txt
│   └── test_predictions.csv
│
├── scrape_news.py
└── README.md


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
