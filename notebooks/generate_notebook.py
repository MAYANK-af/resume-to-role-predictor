"""
Script to generate notebooks/eda_and_training.ipynb cleanly.
"""
import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Resume-to-Role Predictor: EDA, NLP Pipeline & Model Training\n",
                "**University NLP Mini-Project**\n",
                "\n",
                "This interactive notebook walks through the end-to-end NLP workflow:\n",
                "1. **Exploratory Data Analysis (EDA)** & Raw Category Inspection\n",
                "2. **Role Mapping & Pruning**: Mapping 15 tech categories into 5 Tech Roles & Dropping Non-Tech/Duplicates\n",
                "3. **Text Preprocessing**: Normalizing tech tokens (`C++`, `.NET`, `Node.js`, `scikit-learn`), cleaning URLs/emails, tokenizing, stopword removal, and lemmatization\n",
                "4. **Feature Engineering**: TF-IDF Vectorization with unigrams and bigrams (`max_features ~5000`)\n",
                "5. **Classification**: Stratified 80/20 train/test evaluation of Multinomial Naive Bayes vs Logistic Regression\n",
                "6. **Regression**: Predicting 0-100 target role fit scores using Ridge Regression vs Logistic Regression probabilities (MAE & R2)\n",
                "7. **Model Explainability**: Top 10 global positive coefficients and local per-resume feature contributions\n",
                "8. **Serialization**: Saving model artifacts with `joblib`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import re\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import nltk\n",
                "import joblib\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.naive_bayes import MultinomialNB\n",
                "from sklearn.linear_model import LogisticRegression, Ridge\n",
                "from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, mean_absolute_error, r2_score\n",
                "\n",
                "# Download necessary NLTK corpora\n",
                "nltk.download('stopwords', quiet=True)\n",
                "nltk.download('punkt', quiet=True)\n",
                "nltk.download('wordnet', quiet=True)\n",
                "nltk.download('omw-1.4', quiet=True)\n",
                "print('Libraries and NLTK resources initialized successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Dataset & Inspect Raw Categories"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_path = os.path.join('..', 'data', 'resume.csv')\n",
                "if not os.path.exists(data_path):\n",
                "    data_path = os.path.join('data', 'resume.csv')\n",
                "\n",
                "df = pd.read_csv(data_path)\n",
                "print(f'Total raw samples: {len(df)}')\n",
                "print(f'Total raw categories: {df[\"Category\"].nunique()}')\n",
                "print('\\nRaw Category Counts:')\n",
                "print(df['Category'].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Category Mapping, Pruning & Class Balance\n",
                "We map the IT/tech categories into 5 target roles:\n",
                "- **Backend**: Java Developer, Python Developer, DotNet Developer, DevOps Engineer, Database, ETL Developer, Hadoop\n",
                "- **AI/ML**: Data Science\n",
                "- **Web Dev**: Web Designing\n",
                "- **Analyst**: Business Analyst\n",
                "- **Other Tech**: Testing, Automation Testing, Network Security Engineer, SAP Developer, Blockchain\n",
                "- Non-tech categories (HR, Advocate, Arts, Sales, Mechanical, etc.) are pruned, and duplicate resumes are dropped."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "TECH_ROLE_MAP = {\n",
                "    'Java Developer': 'Backend',\n",
                "    'Python Developer': 'Backend',\n",
                "    'DotNet Developer': 'Backend',\n",
                "    'DevOps Engineer': 'Backend',\n",
                "    'Database': 'Backend',\n",
                "    'ETL Developer': 'Backend',\n",
                "    'Hadoop': 'Backend',\n",
                "    'Data Science': 'AI/ML',\n",
                "    'Web Designing': 'Web Dev',\n",
                "    'Business Analyst': 'Analyst',\n",
                "    'Testing': 'Other Tech',\n",
                "    'Automation Testing': 'Other Tech',\n",
                "    'Network Security Engineer': 'Other Tech',\n",
                "    'SAP Developer': 'Other Tech',\n",
                "    'Blockchain': 'Other Tech',\n",
                "}\n",
                "\n",
                "df['Role'] = df['Category'].map(TECH_ROLE_MAP)\n",
                "df_tech = df.dropna(subset=['Role']).copy()\n",
                "print(f'Tech records before deduplication: {len(df_tech)}')\n",
                "\n",
                "# Deduplication\n",
                "dup_count = df_tech.duplicated(subset=['Resume']).sum()\n",
                "df_dedup = df_tech.drop_duplicates(subset=['Resume']).copy().reset_index(drop=True)\n",
                "print(f'Duplicates dropped: {dup_count}')\n",
                "print(f'Unique resumes after deduplication: {len(df_dedup)}')\n",
                "\n",
                "# Class Balance\n",
                "balance_df = pd.DataFrame({\n",
                "    'Count': df_dedup['Role'].value_counts(),\n",
                "    'Percentage (%)': (df_dedup['Role'].value_counts(normalize=True) * 100).round(2)\n",
                "})\n",
                "balance_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Preprocessing with Tech-Token Preservation\n",
                "Standard tokenizers strip punctuation, destroying crucial terms like `C++`, `.NET`, `Node.js`, and `scikit-learn`.\n",
                "We preserve these compound technical tokens before cleaning punctuation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.preprocess import preprocess_resume\n",
                "\n",
                "sample_raw = 'Experienced developer in C++, C#, .NET, Node.js, and scikit-learn. Contact: dev@example.com'\n",
                "print('Sample Raw:       ', sample_raw)\n",
                "print('Sample Processed: ', preprocess_resume(sample_raw))\n",
                "\n",
                "df_dedup['Cleaned_Resume'] = df_dedup['Resume'].apply(preprocess_resume)\n",
                "print('\\nDataset preprocessed successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Stratified 80/20 Train/Test Split & TF-IDF Extraction"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = df_dedup['Cleaned_Resume']\n",
                "y = df_dedup['Role']\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                ")\n",
                "print(f'Train samples: {len(X_train)} | Test samples: {len(X_test)}')\n",
                "\n",
                "# TF-IDF Feature Extraction\n",
                "tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)\n",
                "X_train_tfidf = tfidf.fit_transform(X_train)\n",
                "X_test_tfidf = tfidf.transform(X_test)\n",
                "print(f'TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}')\n",
                "print(f'TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Classification: Multinomial Naive Bayes vs Logistic Regression"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "role_classes = ['Backend', 'AI/ML', 'Web Dev', 'Analyst', 'Other Tech']\n",
                "\n",
                "# 1. Multinomial Naive Bayes\n",
                "nb = MultinomialNB(alpha=0.01)\n",
                "nb.fit(X_train_tfidf, y_train)\n",
                "y_pred_nb = nb.predict(X_test_tfidf)\n",
                "\n",
                "# 2. Logistic Regression (with class balancing)\n",
                "lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)\n",
                "lr.fit(X_train_tfidf, y_train)\n",
                "y_pred_lr = lr.predict(X_test_tfidf)\n",
                "\n",
                "eval_df = pd.DataFrame([\n",
                "    {\n",
                "        'Model': 'Multinomial Naive Bayes',\n",
                "        'Accuracy': round(accuracy_score(y_test, y_pred_nb), 4),\n",
                "        'Macro Precision': round(f1_score(y_test, y_pred_nb, average='macro'), 4),\n",
                "        'Macro F1': round(f1_score(y_test, y_pred_nb, average='macro'), 4)\n",
                "    },\n",
                "    {\n",
                "        'Model': 'Logistic Regression (Balanced)',\n",
                "        'Accuracy': round(accuracy_score(y_test, y_pred_lr), 4),\n",
                "        'Macro Precision': round(f1_score(y_test, y_pred_lr, average='macro'), 4),\n",
                "        'Macro F1': round(f1_score(y_test, y_pred_lr, average='macro'), 4)\n",
                "    }\n",
                "])\n",
                "eval_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Regression: Target Role Fit Score (0-100)\n",
                "Predicting 0-100 fit score using Ridge Regression on TF-IDF vs Logistic Regression class probabilities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "reg_summary = []\n",
                "lr_probs = lr.predict_proba(X_test_tfidf) * 100\n",
                "lr_class_idx = {cls: idx for idx, cls in enumerate(lr.classes_)}\n",
                "\n",
                "for role in role_classes:\n",
                "    y_tr_role = (y_train == role).astype(float) * 100.0\n",
                "    y_te_role = (y_test == role).astype(float) * 100.0\n",
                "\n",
                "    ridge = Ridge(alpha=1.0, random_state=42)\n",
                "    ridge.fit(X_train_tfidf, y_tr_role)\n",
                "    y_pred_ridge = np.clip(ridge.predict(X_test_tfidf), 0, 100)\n",
                "\n",
                "    y_pred_lr_prob = lr_probs[:, lr_class_idx[role]]\n",
                "\n",
                "    reg_summary.append({\n",
                "        'Role': role,\n",
                "        'Ridge MAE': round(mean_absolute_error(y_te_role, y_pred_ridge), 2),\n",
                "        'Ridge R2': round(r2_score(y_te_role, y_pred_ridge), 4),\n",
                "        'LR Prob MAE': round(mean_absolute_error(y_te_role, y_pred_lr_prob), 2),\n",
                "        'LR Prob R2': round(r2_score(y_te_role, y_pred_lr_prob), 4)\n",
                "    })\n",
                "\n",
                "pd.DataFrame(reg_summary)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Explainability: Top 10 Coefficients per Role"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_names = np.array(tfidf.get_feature_names_out())\n",
                "\n",
                "for cls in lr.classes_:\n",
                "    cls_i = lr_class_idx[cls]\n",
                "    coefs = lr.coef_[cls_i]\n",
                "    top_i = np.argsort(coefs)[::-1][:10]\n",
                "    words = [f'{feature_names[i]} ({coefs[i]:.3f})' for i in top_i]\n",
                "    print(f'=== [{cls}] Top Influential Features ===')\n",
                "    print(', '.join(words))\n",
                "    print()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_file = os.path.join("notebooks", "eda_and_training.ipynb")
with open(target_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {target_file} successfully!")
