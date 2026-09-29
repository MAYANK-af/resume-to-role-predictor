"""
src/predict.py
Inference and Explainability module for Resume-to-Role Predictor.
Loads trained models and vectorizer, handles validation, role prediction,
target role fit scoring (Ridge vs LR Probability), and local word-level explainability.
"""

import os
import joblib
import numpy as np
from src.preprocess import preprocess_resume, format_token_display, clean_raw_text

MODELS_DIR = "models"


class ResumePredictor:
    """Encapsulates model loading, validation, prediction, scoring, and explainability."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
        self.lr_model = joblib.load(os.path.join(models_dir, "lr_classifier.joblib"))
        self.nb_model = joblib.load(os.path.join(models_dir, "nb_classifier.joblib"))
        self.ridge_models = joblib.load(os.path.join(models_dir, "ridge_models.joblib"))
        self.metadata = joblib.load(os.path.join(models_dir, "metadata.joblib"))

        self.roles = self.metadata['role_order']
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.lr_class_to_idx = {cls: idx for idx, cls in enumerate(self.lr_model.classes_)}

    def validate_input(self, text: str) -> tuple[bool, str]:
        """Validates that input is non-empty and has sufficient technical context."""
        if not text or not text.strip():
            return False, "Please enter or paste resume text to begin analysis."
        words = text.strip().split()
        if len(words) < 20:
            return False, (
                f"Input resume text is too brief ({len(words)} words). "
                "Please provide a more comprehensive resume (at least 20 words) for reliable role prediction."
            )
        return True, ""

    def predict(self, raw_text: str, target_role: str = None) -> dict:
        """
        Runs complete prediction pipeline on raw resume text.
        Returns:
            - predicted_role (LR)
            - nb_predicted_role (NB)
            - probabilities (dict of role -> percentage)
            - target_role
            - ridge_fit_score (0-100)
            - lr_prob_fit_score (0-100)
            - top_driving_words (list of {display, token, contribution, tfidf})
            - target_driving_words (list for chosen target role)
            - cleaned_tokens (list)
        """
        is_valid, msg = self.validate_input(raw_text)
        if not is_valid:
            raise ValueError(msg)

        # 1. Preprocess
        cleaned_text = preprocess_resume(raw_text)
        tokens = cleaned_text.split()

        # 2. Vectorize
        tfidf_vec = self.vectorizer.transform([cleaned_text])

        # 3. Classify
        predicted_role = self.lr_model.predict(tfidf_vec)[0]
        nb_predicted_role = self.nb_model.predict(tfidf_vec)[0]

        # Probabilities
        lr_probs = self.lr_model.predict_proba(tfidf_vec)[0]
        probabilities = {
            role: round(float(lr_probs[self.lr_class_to_idx[role]] * 100), 1)
            for role in self.roles
        }

        # If no target role or invalid, default to predicted role
        if not target_role or target_role not in self.roles:
            target_role = predicted_role

        # 4. Regression Fit Score (0-100)
        ridge_fit_score = float(np.clip(self.ridge_models[target_role].predict(tfidf_vec)[0], 0.0, 100.0))
        lr_prob_fit_score = float(probabilities[target_role])

        # 5. Local Explainability (Feature Contribution = TF-IDF * Coefficient)
        predicted_words = self._compute_local_contributions(tfidf_vec, predicted_role)
        target_words = self._compute_local_contributions(tfidf_vec, target_role)

        return {
            'predicted_role': predicted_role,
            'nb_predicted_role': nb_predicted_role,
            'probabilities': probabilities,
            'target_role': target_role,
            'ridge_fit_score': round(ridge_fit_score, 1),
            'lr_prob_fit_score': round(lr_prob_fit_score, 1),
            'top_driving_words': predicted_words,
            'target_driving_words': target_words,
            'cleaned_token_count': len(tokens),
            'cleaned_tokens': tokens,
        }

    def _compute_local_contributions(self, tfidf_vec, role: str) -> list[dict]:
        """Calculates per-input word contributions for a given role: tfidf * lr_coef."""
        role_idx = self.lr_class_to_idx[role]
        coefs = self.lr_model.coef_[role_idx]

        # Extract non-zero indices in input TF-IDF vector
        row, cols = tfidf_vec.nonzero()
        contributions = []

        for c in cols:
            tfidf_val = tfidf_vec[0, c]
            coef = coefs[c]
            contrib = tfidf_val * coef

            if contrib > 0.001:  # Only positive driving contributions
                raw_token = self.feature_names[c]
                display_name = format_token_display(raw_token)
                contributions.append({
                    'token': raw_token,
                    'display': display_name,
                    'contribution': round(float(contrib), 4),
                    'tfidf': round(float(tfidf_val), 4),
                    'coef': round(float(coef), 4)
                })

        # Sort descending by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        return contributions[:12]  # Return top 12 keywords


# Singleton instance helper
_PREDICTOR_INSTANCE = None


def get_predictor() -> ResumePredictor:
    """Returns a cached ResumePredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ResumePredictor()
    return _PREDICTOR_INSTANCE
