"""
server.py
FastAPI backend wrapper for Resume-to-Role Predictor.
Keeps existing model code untouched. Exposes POST /predict and metadata endpoints.
"""

import os
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import get_predictor

app = FastAPI(
    title="Resume-to-Role Predictor API",
    description="3D Word Constellation & Role Alignment Engine",
    version="2.0.0"
)

# Enable CORS for local Vite development and preview servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load model predictor
predictor = get_predictor()


class PredictRequest(BaseModel):
    resume_text: Optional[str] = Field(None, description="Raw text of the resume to analyze")
    text: Optional[str] = Field(None, description="Alias for resume_text")
    target_role: Optional[str] = Field(None, description="Optional target role to score against")

    def get_text(self) -> str:
        return self.resume_text or self.text or ""


class WordWeight(BaseModel):
    word: str
    weight: float
    tfidf: Optional[float] = None
    coef: Optional[float] = None


class PredictResponse(BaseModel):
    predicted_role: str
    nb_predicted_role: str
    target_role: str
    fit_score: float
    lr_prob_fit_score: float
    probabilities: Dict[str, float]
    top_words: List[WordWeight]
    target_words: List[WordWeight]
    cleaned_tokens: List[str]


# Pre-packaged editorial sample profiles
SAMPLE_PROFILES = {
    "AI / ML": (
        "Senior Data Scientist and Machine Learning Engineer with 5+ years of experience building and deploying "
        "NLP and deep learning systems. Proficient in Python, pandas, numpy, scikit-learn, PyTorch, and TensorFlow. "
        "Developed transformer-based text classification pipelines, clustering algorithms, and recommendation engines. "
        "Expertise in exploratory data analysis, feature engineering, regression modeling, hyperparameter tuning, "
        "model explainability with SHAP, and SQL for querying relational databases. Designed automated ETL pipelines "
        "and monitored predictive ML models in production environments."
    ),
    "Backend": (
        "Senior Backend Developer with 6 years of experience architecting high-performance enterprise applications "
        "and distributed microservices. Core expertise in Java, Spring Boot, Hibernate, Python, and SQL Server. "
        "Designed and implemented RESTful APIs, optimized complex database queries in PostgreSQL and Oracle, "
        "and integrated Redis caching layers. Extensive experience in Docker, Kubernetes, CI/CD pipelines with Jenkins "
        "and GitLab, and messaging queues using Apache Kafka and RabbitMQ. Strong background in database design and OOP."
    ),
    "Web Dev": (
        "Creative Front-End Web Designer and Developer with 4 years of experience crafting modern, responsive web user interfaces. "
        "Proficient in HTML5, CSS3, JavaScript, jQuery, Bootstrap, React.js, and WordPress theme customization. "
        "Skilled in Adobe Photoshop, Figma, graphic design, responsive grid layouts, and cross-browser compatibility. "
        "Developed client websites, optimized UI/UX design workflows, and integrated REST APIs. Strong knowledge of web "
        "accessibility standards, animation effects, and mobile-first responsive design."
    ),
    "Analyst": (
        "Business Analyst with 5 years experience driving business intelligence, requirement gathering, and process re-engineering. "
        "Expert in translating complex business stakeholder requirements into functional specifications, user stories, and wireframes. "
        "Proficient in SQL, Excel advanced financial modeling, Tableau, Power BI, and ERP systems. "
        "Conducted GAP analysis, cost-benefit analysis, executive reporting, and KPI dashboard development. "
        "Facilitated sprint planning, client requirement workshops, and user acceptance testing."
    ),
    "Other Tech": (
        "QA Automation Engineer and Software Tester with 5 years experience in manual and automated testing of web applications. "
        "Proficient in Selenium WebDriver, TestNG, Java, Python, and Postman API testing. Built robust test automation frameworks, "
        "executed regression testing suites, performance testing, and bug tracking using JIRA. Experience in network security protocols, "
        "vulnerability scanning, and blockchain smart contract testing. Certified ISTQB software quality engineer."
    )
}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Resume-to-Role Predictor API"}


@app.get("/metadata")
def get_metadata():
    """Returns role taxonomy and global top feature coefficients for interactive 3D inspection."""
    meta = predictor.metadata
    return {
        "roles": predictor.roles,
        "top_coefficients": meta.get("top_coefficients", {}),
        "comparison_metrics": meta.get("comparison_df", {}).to_dict(orient="records") if hasattr(meta.get("comparison_df"), "to_dict") else meta.get("comparison_df"),
        "regression_comparison": meta.get("regression_comparison", {}).to_dict(orient="records") if hasattr(meta.get("regression_comparison"), "to_dict") else meta.get("regression_comparison")
    }


@app.get("/samples")
def get_samples():
    """Returns realistic curated technical resumes for zero-friction user testing."""
    return SAMPLE_PROFILES


@app.post("/predict", response_model=PredictResponse)
def predict_resume(request: PredictRequest):
    """
    Analyzes resume text:
    - Validates minimum word length
    - Classifies best-fit tech role
    - Computes 0-100 fit score (Ridge Regression)
    - Returns per-role probabilities and top influential word contributions
    """
    input_text = request.get_text()
    is_valid, err_msg = predictor.validate_input(input_text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    try:
        raw_res = predictor.predict(input_text, target_role=request.target_role)

        # Map top words to {word, weight} as required by prompt
        top_words = [
            WordWeight(
                word=w["display"],
                weight=w["contribution"],
                tfidf=w.get("tfidf"),
                coef=w.get("coef")
            )
            for w in raw_res["top_driving_words"]
        ]

        target_words = [
            WordWeight(
                word=w["display"],
                weight=w["contribution"],
                tfidf=w.get("tfidf"),
                coef=w.get("coef")
            )
            for w in raw_res["target_driving_words"]
        ]

        return PredictResponse(
            predicted_role=raw_res["predicted_role"],
            nb_predicted_role=raw_res["nb_predicted_role"],
            target_role=raw_res["target_role"],
            fit_score=raw_res["ridge_fit_score"],
            lr_prob_fit_score=raw_res["lr_prob_fit_score"],
            probabilities=raw_res["probabilities"],
            top_words=top_words,
            target_words=target_words,
            cleaned_tokens=raw_res["cleaned_tokens"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount compiled 3D React frontend if built
dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

