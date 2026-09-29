"""
src/train.py
Model training, evaluation, and serialization for Resume-to-Role Predictor.
Trains MultinomialNB and Logistic Regression classifiers,
trains Ridge Regression for 0-100 role fit scoring,
computes evaluation metrics (Accuracy, Macro-F1, Confusion Matrix, MAE, R2),
and extracts top coefficients for model explainability.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    r2_score
)

from src.preprocess import preprocess_resume, format_token_display

DATA_PATH = os.path.join("data", "resume.csv")
MODELS_DIR = "models"

# 5 Target Roles Mapping
TECH_ROLE_MAP = {
    'Java Developer': 'Backend',
    'Python Developer': 'Backend',
    'DotNet Developer': 'Backend',
    'DevOps Engineer': 'Backend',
    'Database': 'Backend',
    'ETL Developer': 'Backend',
    'Hadoop': 'Backend',
    'Data Science': 'AI/ML',
    'Web Designing': 'Web Dev',
    'Business Analyst': 'Analyst',
    'Testing': 'Other Tech',
    'Automation Testing': 'Other Tech',
    'Network Security Engineer': 'Other Tech',
    'SAP Developer': 'Other Tech',
    'Blockchain': 'Other Tech',
}

ROLE_ORDER = ['Backend', 'AI/ML', 'Web Dev', 'Analyst', 'Other Tech']


def acquire_dataset() -> pd.DataFrame:
    """
    Attempts to load data/resume.csv.
    If not found, attempts kagglehub download.
    If kagglehub fails, raises a clear RuntimeError instructing the user.
    """
    if os.path.exists(DATA_PATH):
        print(f"[DATA] Loading dataset from '{DATA_PATH}'...")
        return pd.read_csv(DATA_PATH)

    print(f"[DATA] '{DATA_PATH}' not found. Attempting download via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("jillanisofttech/updated-resume-dataset")
        csv_candidates = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_candidates:
            source_file = os.path.join(path, csv_candidates[0])
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            import shutil
            shutil.copy(source_file, DATA_PATH)
            print(f"[DATA] Downloaded and copied to '{DATA_PATH}'.")
            return pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] kagglehub download failed: {e}")

    # Fallback instruction as required by prompt
    raise RuntimeError(
        f"Kaggle download failed or credentials missing.\n"
        f"Please manually place the Kaggle Resume Dataset CSV at '{DATA_PATH}'."
    )


def prepare_data(df: pd.DataFrame):
    """
    Inspects category labels, filters tech categories, maps them into 5 roles,
    removes duplicates, and checks class balance.
    """
    print("\n--- 1. Raw Category Inspection ---")
    print(f"Total raw records: {len(df)}")
    print(f"Total unique raw categories: {df['Category'].nunique()}")
    print("Raw Category Counts:")
    print(df['Category'].value_counts())

    # Map to 5 roles
    df['Role'] = df['Category'].map(TECH_ROLE_MAP)

    non_tech_dropped = df['Role'].isna().sum()
    print(f"\nNon-tech records dropped: {non_tech_dropped}")

    df_tech = df.dropna(subset=['Role']).copy()
    print(f"Tech records before deduplication: {len(df_tech)}")
    print(df_tech['Role'].value_counts())

    # Remove duplicates
    dup_count = df_tech.duplicated(subset=['Resume']).sum()
    df_dedup = df_tech.drop_duplicates(subset=['Resume']).copy().reset_index(drop=True)
    print(f"\nDuplicates removed: {dup_count}")
    print(f"Unique tech resumes: {len(df_dedup)}")

    print("\n--- Class Balance (Deduplicated) ---")
    balance_df = pd.DataFrame({
        'Count': df_dedup['Role'].value_counts(),
        'Percentage': (df_dedup['Role'].value_counts(normalize=True) * 100).round(2)
    })
    print(balance_df)

    return df_dedup, balance_df


def train_and_evaluate():
    """Main training, evaluation, and export pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    df = acquire_dataset()
    df_clean, balance_df = prepare_data(df)

    # 2. Text Preprocessing
    print("\n--- 2. Preprocessing Resume Texts ---")
    df_clean['Cleaned_Resume'] = df_clean['Resume'].apply(preprocess_resume)
    print("Sample cleaned resume text snippet:")
    print(df_clean['Cleaned_Resume'].iloc[0][:300], "...\n")

    # 3. Stratified 80/20 Split
    print("--- 3. Train/Test Stratified Split (80/20) ---")
    X = df_clean['Cleaned_Resume']
    y = df_clean['Role']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("Train Role Distribution:\n", y_train.value_counts())
    print("Test Role Distribution:\n", y_test.value_counts())

    # 4. Feature Extraction: TF-IDF
    print("\n--- 4. TF-IDF Feature Extraction ---")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    print(f"TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")

    # 5. Classification: Multinomial Naive Bayes vs Logistic Regression
    print("\n--- 5. Training Classification Models ---")

    # Naive Bayes (alpha=0.01 for better smoothing with sparse TF-IDF)
    nb = MultinomialNB(alpha=0.01)
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)

    # Logistic Regression (with class_weight='balanced' to handle role imbalance)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)

    # Calculate metrics
    classes = ROLE_ORDER

    def compute_metrics(y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        return {
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Macro Precision': round(prec, 4),
            'Macro Recall': round(rec, 4),
            'Macro F1': round(f1, 4),
            'Confusion Matrix': cm
        }

    nb_metrics = compute_metrics(y_test, y_pred_nb, "Multinomial Naive Bayes")
    lr_metrics = compute_metrics(y_test, y_pred_lr, "Logistic Regression")

    comparison_df = pd.DataFrame([
        {k: v for k, v in nb_metrics.items() if k != 'Confusion Matrix'},
        {k: v for k, v in lr_metrics.items() if k != 'Confusion Matrix'}
    ])

    print("\n--- Model Comparison Table ---")
    print(comparison_df.to_string(index=False))

    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr, labels=classes, zero_division=0))

    # 6. Regression: Target Role Fit Score (0-100) using Ridge Regression vs LR Probabilities
    print("\n--- 6. Target Role Fit Score: Ridge Regression vs LR Class Probabilities ---")
    ridge_models = {}
    regression_comparison = []

    # LR test probabilities (scaled to 0-100)
    lr_probs = lr.predict_proba(X_test_tfidf) * 100
    lr_class_to_idx = {cls: idx for idx, cls in enumerate(lr.classes_)}

    for role in classes:
        # Binary target scaled to 100 (100 if role, 0 otherwise)
        y_train_role = (y_train == role).astype(float) * 100.0
        y_test_role = (y_test == role).astype(float) * 100.0

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_tfidf, y_train_role)
        ridge_models[role] = ridge

        # Predictions clipped to [0, 100]
        y_pred_ridge = np.clip(ridge.predict(X_test_tfidf), 0, 100)

        # LR probability for this role
        role_lr_idx = lr_class_to_idx[role]
        y_pred_lr_prob = lr_probs[:, role_lr_idx]

        # Compute MAE and R2 for Ridge
        ridge_mae = mean_absolute_error(y_test_role, y_pred_ridge)
        ridge_r2 = r2_score(y_test_role, y_pred_ridge)

        # Compute MAE and R2 for LR Probability
        lr_mae = mean_absolute_error(y_test_role, y_pred_lr_prob)
        lr_r2 = r2_score(y_test_role, y_pred_lr_prob)

        regression_comparison.append({
            'Target Role': role,
            'Ridge MAE': round(ridge_mae, 2),
            'Ridge R2': round(ridge_r2, 4),
            'LR Prob MAE': round(lr_mae, 2),
            'LR Prob R2': round(lr_r2, 4),
        })

    reg_comp_df = pd.DataFrame(regression_comparison)
    print("\n--- Regression Fit Score Evaluation (0-100 scale on Test Set) ---")
    print(reg_comp_df.to_string(index=False))

    # 7. Explainability: Top 10 Positive Coefficients per Role
    print("\n--- 7. Top 10 Influential Feature Words per Role (Logistic Regression) ---")
    feature_names = np.array(tfidf.get_feature_names_out())
    top_coefficients = {}

    for cls in lr.classes_:
        cls_idx = lr_class_to_idx[cls]
        coefs = lr.coef_[cls_idx]
        top_positive_indices = np.argsort(coefs)[::-1][:10]

        top_words = []
        for idx in top_positive_indices:
            raw_token = feature_names[idx]
            display_name = format_token_display(raw_token)
            top_words.append({
                'token': raw_token,
                'display': display_name,
                'weight': round(float(coefs[idx]), 4)
            })
        top_coefficients[cls] = top_words

        words_str = ", ".join([f"{w['display']} ({w['weight']})" for w in top_words])
        print(f"\n[{cls}]:")
        print(f"  {words_str}")

    # 8. Save Artifacts with joblib
    print("\n--- 8. Saving Model Artifacts ---")
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    joblib.dump(nb, os.path.join(MODELS_DIR, "nb_classifier.joblib"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "lr_classifier.joblib"))
    joblib.dump(ridge_models, os.path.join(MODELS_DIR, "ridge_models.joblib"))

    # Save comprehensive metadata
    metadata = {
        'roles': classes,
        'role_order': ROLE_ORDER,
        'comparison_df': comparison_df,
        'regression_comparison': reg_comp_df,
        'nb_metrics': nb_metrics,
        'lr_metrics': lr_metrics,
        'top_coefficients': top_coefficients,
        'lr_classes': list(lr.classes_),
        'class_balance': balance_df.to_dict(),
        'test_size': len(X_test),
        'train_size': len(X_train),
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.joblib"))

    # Also save confusion matrix plots for Streamlit visualization
    save_confusion_matrix_plot(lr_metrics['Confusion Matrix'], classes, "lr_confusion_matrix.png", "Logistic Regression")
    save_confusion_matrix_plot(nb_metrics['Confusion Matrix'], classes, "nb_confusion_matrix.png", "Multinomial Naive Bayes")

    print(f"\nAll models, vectorizer, and metadata saved to '{MODELS_DIR}/'.")
    print("Training pipeline completed successfully!")


def save_confusion_matrix_plot(cm, classes, filename, model_name="Classifier"):
    """Saves a dark-mode styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0b0f19')
    ax.set_facecolor('#0b0f19')

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{model_name} Confusion Matrix", color='#e2e8f0', fontsize=13, pad=15)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right', color='#94a3b8', fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, color='#94a3b8', fontsize=10)

    # Label values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, 'd'),
                    ha="center", va="center",
                    color="white" if val > thresh else "#cbd5e1",
                    fontweight="bold")

    ax.set_ylabel('True Role', color='#e2e8f0', fontsize=11)
    ax.set_xlabel('Predicted Role', color='#e2e8f0', fontsize=11)

    for spine in ax.spines.values():
        spine.set_color('#1e293b')

    plt.tight_layout()
    output_path = os.path.join(MODELS_DIR, filename)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    train_and_evaluate()
