# ATS Resume Matcher — NLP-Powered

A Flask web app that analyzes resumes using **TF-IDF vectorization** and **cosine similarity** to match against job descriptions and provide ATS scoring.

---

## Setup

```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

---

## How TF-IDF Matching Works

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into numerical vectors:
- **TF**: How often a word appears in a document
- **IDF**: How rare that word is across documents (rare = more important)

**Cosine Similarity** then measures the angle between the resume vector and job description vector:
- `1.0` = identical content
- `0.0` = completely different
- `0.3–0.6` = moderate match (typical for a decent resume)

---

## Scoring Formula

| Mode | Formula |
|------|---------|
| Without JD | `heuristic_score / 60 × 100` |
| With JD | `heuristic × 60% + TF-IDF × 40%` |

### Heuristic breakdown (60 pts total):
| Category | Max |
|----------|-----|
| Sections detected | 20 |
| Contact info | 12 |
| Metrics in bullets | 14 |
| Action verbs | 8 |
| Resume length | 6 |

---

## Resume-Worthy Bullet Points for This Project

- Built a Flask-based NLP resume analyzer using TF-IDF vectorization and cosine similarity to compute ATS match scores against job descriptions
- Implemented skill overlap detection across 50+ tech skills, identifying matching, missing, and extra skills between resume and JD
- Engineered a hybrid scoring system combining heuristic ATS analysis (sections, metrics, verbs) with TF-IDF similarity for accurate job fit scoring
- Developed keyword gap analysis using scikit-learn TfidfVectorizer to surface missing JD terms from candidate resumes

---

## File Structure

```
resume-analyzer/
├── app.py              # Flask backend + NLP logic
├── requirements.txt
├── README.md
└── templates/
    └── index.html      # Frontend (dark minimal UI)
```

---

## Tech Stack
- Python, Flask, scikit-learn, pdfplumber, python-docx
- TF-IDF via `TfidfVectorizer`, similarity via `cosine_similarity`
- Vanilla HTML/CSS/JS frontend — no React, no build step
