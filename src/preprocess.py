"""
src/preprocess.py
Text Preprocessing module for Resume-to-Role Predictor.
Handles URL/email removal, tech-token preservation, punctuation cleaning,
stopword removal, and lemmatization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK corpora are downloaded
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}' if 'wordnet' in resource or 'stopwords' in resource or 'omw' in resource else f'tokenizers/{resource}')
    except (LookupError, IndexError):
        nltk.download(resource, quiet=True)

# Tech token mappings to preserve during punctuation cleaning
TECH_TOKEN_RULES = [
    (r'(?i)(?<![a-z0-9])c\+\+(?![a-z0-9])', ' cplusplus '),
    (r'(?i)(?<![a-z0-9])c#(?![a-z0-9])', ' csharp '),
    (r'(?i)(?<![a-z0-9])asp\.net(?![a-z0-9])', ' aspdotnet '),
    (r'(?i)(?<![a-z0-9])\.net(?![a-z0-9])', ' dotnet '),
    (r'(?i)(?<![a-z0-9])node\.js(?![a-z0-9])', ' nodejs '),
    (r'(?i)(?<![a-z0-9])react\.js(?![a-z0-9])', ' reactjs '),
    (r'(?i)(?<![a-z0-9])vue\.js(?![a-z0-9])', ' vuejs '),
    (r'(?i)(?<![a-z0-9])next\.js(?![a-z0-9])', ' nextjs '),
    (r'(?i)(?<![a-z0-9])angular\.js(?![a-z0-9])', ' angularjs '),
    (r'(?i)(?<![a-z0-9])scikit-learn(?![a-z0-9])', ' scikitlearn '),
    (r'(?i)(?<![a-z0-9])ci/cd(?![a-z0-9])', ' cicd '),
    (r'(?i)(?<![a-z0-9])pl/sql(?![a-z0-9])', ' plsql '),
    (r'(?i)(?<![a-z0-9])t-sql(?![a-z0-9])', ' tsql '),
    (r'(?i)(?<![a-z0-9])tcp/ip(?![a-z0-9])', ' tcpip '),
    (r'(?i)(?<![a-z0-9])k-means(?![a-z0-9])', ' kmeans '),
    (r'(?i)(?<![a-z0-9])r&d(?![a-z0-9])', ' rnd '),
]

# Display map for explainability (normalized -> display name)
TOKEN_DISPLAY_MAP = {
    'cplusplus': 'C++',
    'csharp': 'C#',
    'dotnet': '.NET',
    'aspdotnet': 'ASP.NET',
    'nodejs': 'Node.js',
    'reactjs': 'React.js',
    'vuejs': 'Vue.js',
    'nextjs': 'Next.js',
    'angularjs': 'Angular.js',
    'scikitlearn': 'scikit-learn',
    'cicd': 'CI/CD',
    'plsql': 'PL/SQL',
    'tsql': 'T-SQL',
    'tcpip': 'TCP/IP',
    'kmeans': 'k-means',
    'rnd': 'R&D',
}

_STOP_WORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()


def normalize_tech_tokens(text: str) -> str:
    """Replaces tech-specific symbols (e.g., C++, .NET, Node.js) with unified token words."""
    for pattern, replacement in TECH_TOKEN_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def clean_raw_text(text: str) -> str:
    """Removes URLs, emails, and extraneous characters."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Preserve tech tokens
    text = normalize_tech_tokens(text)
    # Lowercase
    text = text.lower()
    # Replace non-alphanumeric (except single space)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Tokenizes text, strips stopwords, and applies WordNet lemmatization."""
    cleaned = clean_raw_text(text)
    if not cleaned:
        return []
    tokens = nltk.word_tokenize(cleaned)
    processed = []
    for token in tokens:
        if token not in _STOP_WORDS and len(token) > 1 and not token.isdigit():
            lemma = _LEMMATIZER.lemmatize(token)
            processed.append(lemma)
    return processed


def preprocess_resume(text: str) -> str:
    """
    Full preprocessing pipeline:
    Input: raw resume string
    Output: space-separated lemmatized tokens string ready for TF-IDF.
    """
    tokens = tokenize_and_lemmatize(text)
    return " ".join(tokens)


def format_token_display(token: str) -> str:
    """Formats a token into human-friendly format using the display map or title case."""
    if token in TOKEN_DISPLAY_MAP:
        return TOKEN_DISPLAY_MAP[token]
    # Handle bigrams if any
    parts = token.split()
    if len(parts) > 1:
        return " ".join(TOKEN_DISPLAY_MAP.get(p, p) for p in parts)
    return token
