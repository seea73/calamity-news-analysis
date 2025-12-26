import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Try to import matplotlib for confusion matrix plots (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "data", "processed", "news_labeled.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading dataset:", input_path)
    df = pd.read_csv(input_path)

    # -----------------------------
    # Filter & prepare data
    # -----------------------------
    print("Initial rows:", len(df))

    # Remove "unknown" labels
    df = df[df["calamity_label"] != "unknown"].copy()
    df = df.dropna(subset=["text_lemma"])
    print("Rows after removing 'unknown' and NaNs:", len(df))

    X = df["text_lemma"].astype(str)
    y = df["calamity_label"].astype(str)

    print("\nLabel distribution:")
    print(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    # -----------------------------
    # TF-IDF vectorization
    # -----------------------------
    print("\nVectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)  # unigrams + bigrams
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -----------------------------
    # Model 1: Multinomial Naive Bayes
    # -----------------------------
    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    y_pred_nb = nb_model.predict(X_test_tfidf)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb)

    print("\n=== Naive Bayes Results ===")
    print("Accuracy:", acc_nb)
    print(report_nb)

    # -----------------------------
    # Model 2: Logistic Regression
    # -----------------------------
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=300)
    lr_model.fit(X_train_tfidf, y_train)

    y_pred_lr = lr_model.predict(X_test_tfidf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)

    print("\n=== Logistic Regression Results ===")
    print("Accuracy:", acc_lr)
    print(report_lr)

    # -----------------------------
    # Save reports to a text file
    # -----------------------------
    report_path = os.path.join(results_dir, "model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Naive Bayes ===\n")
        f.write(f"Accuracy: {acc_nb:.4f}\n\n")
        f.write(report_nb)
        f.write("\n\n=== Logistic Regression ===\n")
        f.write(f"Accuracy: {acc_lr:.4f}\n\n")
        f.write(report_lr)

    print("\nSaved model report →", report_path)

    # -----------------------------
    # Save predictions to CSV
    # -----------------------------
    preds_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "pred_nb": y_pred_nb,
        "pred_lr": y_pred_lr
    })

    preds_path = os.path.join(results_dir, "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print("Saved test predictions →", preds_path)

    # -----------------------------
    # Optional: Confusion matrices
    # -----------------------------
    if MATPLOTLIB_AVAILABLE:
        labels = sorted(y.unique())

        # Naive Bayes confusion matrix
        cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_nb, interpolation="nearest")
        plt.title("Naive Bayes Confusion Matrix")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.colorbar()
        plt.tight_layout()
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm_nb[i, j], ha="center", va="center")
        cm_nb_path = os.path.join(results_dir, "confusion_matrix_nb.png")
        plt.savefig(cm_nb_path)
        plt.close()

        # Logistic Regression confusion matrix
        cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_lr, interpolation="nearest")
        plt.title("Logistic Regression Confusion Matrix")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.colorbar()
        plt.tight_layout()
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm_lr[i, j], ha="center", va="center")
        cm_lr_path = os.path.join(results_dir, "confusion_matrix_lr.png")
        plt.savefig(cm_lr_path)
        plt.close()

        print("Saved confusion matrices →")
        print(" ", cm_nb_path)
        print(" ", cm_lr_path)
    else:
        print("\nmatplotlib not installed, skipping confusion matrix plots.")
        print("You can install it with: pip install matplotlib")


if __name__ == "__main__":
    main()
