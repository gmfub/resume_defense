# utils/analysis.py
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Stemmer (run -> running)
stemmer = PorterStemmer()

# The "Ignore List" (Teammate can expand this)
STOP_WORDS = {
    "responsibilities",
    "requirements",
    "experience",
    "year",
    "years",
    "work",
    "team",
    "player",
    "communication",
    "skills",
    "including",
    "knowledge",
    "understanding",
    "familiarity",
    "proven",
    "track",
    "record",
    "ability",
    "candidate",
    "person",
    "performance",
    "integration",
    "implement",
    "experienced",
    "strong",
    "demonstrated",
    "working",
    "environment",
    "the",
    "and",
    "is",
    "in",
    "to",
    "of",
    "for",
    "with",
    "a",
    "an",
}


def clean_and_stem(text):
    # 1. Remove non-letters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # 2. Lowercase & Split
    words = text.lower().split()
    # 3. Stemming & Filtering
    clean_words = []
    for word in words:
        if word not in STOP_WORDS and len(word) > 2:
            clean_words.append(stemmer.stem(word))
    return " ".join(clean_words)


def calculate_similarity(resume_text, job_desc):
    # Clean both inputs first
    clean_resume = clean_and_stem(resume_text)
    clean_job = clean_and_stem(job_desc)

    text_list = [clean_resume, clean_job]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)


def find_missing_keywords(resume_text, job_desc):
    resume_stems = set(clean_and_stem(resume_text).split())
    job_stems = set(clean_and_stem(job_desc).split())

    missing_stems = job_stems - resume_stems

    display_missing = []
    original_job_words = re.sub(r"[^a-zA-Z\s]", "", job_desc).lower().split()

    for stem in missing_stems:
        for word in original_job_words:
            if stemmer.stem(word) == stem:
                display_missing.append(word)
                break  # Found one representative, move to next stem

    return sorted(list(set(display_missing)), key=len, reverse=True)[:10]
