from flask import Flask, request, jsonify, render_template
import os
import re
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# text extraction


def extract_text_from_docx(filepath):
    try:
        from docx import Document
        doc = Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs])
    except:
        return ""


def extract_text_from_pdf(filepath):
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return '\n'.join([page.extract_text() or '' for page in pdf.pages])
    except:
        return ""


def extract_text(filepath, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'docx':
        return extract_text_from_docx(filepath)
    if ext == 'pdf':
        return extract_text_from_pdf(filepath)
    if ext == 'txt':
        with open(filepath, 'r', errors='ignore') as f:
            return f.read()
    return ""

# NLP matching


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tfidf_similarity(resume_text, jd_text):
    corpus = [clean_text(resume_text), clean_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score), 4)


def find_skill_overlap(resume_text, jd_text):
    SKILL_SET = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift',
        'html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'fastapi',
        'spring', 'nextjs', 'tailwind', 'bootstrap',
        'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'redis', 'firebase',
        'git', 'github', 'gitlab', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'linux', 'bash',
        'machine learning', 'deep learning', 'nlp', 'natural language processing',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'opencv',
        'rest api', 'graphql', 'microservices', 'agile', 'scrum', 'ci/cd', 'devops',
        'data structures', 'algorithms', 'oop', 'system design',
        'tableau', 'power bi', 'matplotlib', 'vercel', 'postman', 'figma',
    ]
    r, j = resume_text.lower(), jd_text.lower()
    matching = [s for s in SKILL_SET if s in r and s in j]
    missing = [s for s in SKILL_SET if s not in r and s in j]
    extra = [s for s in SKILL_SET if s in r and s not in j]
    return matching, missing, extra[:10]


def missing_keywords_from_jd(resume_text, jd_text, top_n=15):
    cleaned_jd = clean_text(jd_text)
    cleaned_resume = clean_text(resume_text)
    try:
        vec = TfidfVectorizer(stop_words='english',
                              ngram_range=(1, 2), max_features=200)
        vec.fit_transform([cleaned_jd])
        jd_terms = vec.get_feature_names_out()
        return [t for t in jd_terms if t not in cleaned_resume][:top_n]
    except:
        return []

# heuristic ATS analysis


SECTIONS = {
    'contact':      ['email', 'phone', 'linkedin', 'github', 'mobile'],
    'education':    ['education', 'university', 'college', 'btech', 'degree', 'gpa', 'cgpa'],
    'experience':   ['experience', 'intern', 'internship', 'work', 'position'],
    'skills':       ['skills', 'technologies', 'programming', 'frameworks', 'tools'],
    'projects':     ['projects', 'project', 'built', 'developed', 'implemented'],
    'achievements': ['achievement', 'award', 'hackathon', 'certification', 'leadership'],
}

WEAK_VERBS = ['worked on', 'helped', 'assisted',
              'responsible for', 'participated in', 'involved in']
ACTION_VERBS = [
    'achieved', 'analyzed', 'architected', 'automated', 'built', 'collaborated', 'created',
    'delivered', 'deployed', 'designed', 'developed', 'engineered', 'enhanced', 'improved',
    'integrated', 'launched', 'led', 'managed', 'optimized', 'reduced', 'scaled', 'solved', 'streamlined',
]


def check_sections(text):
    lower = text.lower()
    found = [s for s, kws in SECTIONS.items() if any(k in lower for k in kws)]
    missing = [s for s in SECTIONS if s not in found]
    return found, missing


def check_contact(text):
    issues, lower = [], text.lower()
    if not re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text):
        issues.append("No email found")
    if not re.search(r'[\+\d][\d\s\-\(\)]{8,}', text):
        issues.append("No phone number")
    if 'linkedin' not in lower:
        issues.append("LinkedIn URL missing")
    if 'github' not in lower:
        issues.append("GitHub URL missing")
    return issues


def check_quantification(text):
    bullets = re.findall(r'[•\-\*]\s*(.+)', text)
    quantified = [b for b in bullets if re.search(r'\d+', b)]
    unquantified = [b for b in bullets if not re.search(r'\d+', b)]
    ratio = len(quantified) / max(len(bullets), 1)
    return ratio, unquantified[:4]


def check_action_verbs(text):
    lines = [l.lstrip('•-* ') for l in text.split('\n') if l.strip()]
    strong = [l for l in lines if l.split() and l.split()[0].lower()
              in ACTION_VERBS]
    weak = [l for l in lines if any(w in l.lower() for w in WEAK_VERBS)]
    return strong, weak


def check_length(text):
    words = len(text.split())
    if words < 200:
        return 'too_short', words
    if words > 900:
        return 'too_long', words
    return 'good', words


def check_cliches(text):
    patterns = [
        (r'\bresponsible for\b', '"Responsible for" — use an action verb'),
        (r'\bteam player\b',     '"Team player" — cliché, remove'),
        (r'\bpassionate\b',      '"Passionate" — show via achievements'),
        (r'\bhard[- ]working\b', '"Hard-working" — cliché, remove'),
    ]
    return [msg for pat, msg in patterns if re.search(pat, text.lower())]


def heuristic_score(sections_found, contact_issues, quant_ratio, strong_verbs, weak_verbs, length_status):
    s = min(20, len(sections_found) * 4)
    c = max(0, 12 - len(contact_issues) * 4)
    q = int(quant_ratio * 14)
    v = max(0, min(8, len(strong_verbs)*2) - min(6, len(weak_verbs)*2))
    l = 6 if length_status == 'good' else 3
    return max(0, min(60, s + c + q + v + l))


def compute_final_score(h_score, similarity, has_jd):
    if not has_jd:
        return round(h_score / 60 * 100)
    return min(100, round(h_score * 0.60) + round(similarity * 100 * 0.40))

# Analysis


def analyze(resume_text, jd_text=''):
    has_jd = bool(jd_text and jd_text.strip())

    sections_found, sections_missing = check_sections(resume_text)
    contact_issues = check_contact(resume_text)
    quant_ratio, unquantified = check_quantification(resume_text)
    strong_verbs, weak_verbs = check_action_verbs(resume_text)
    length_status, word_count = check_length(resume_text)
    cliches = check_cliches(resume_text)

    h_score = heuristic_score(sections_found, contact_issues,
                              quant_ratio, strong_verbs, weak_verbs, length_status)
    similarity = tfidf_similarity(resume_text, jd_text) if has_jd else 0

    matching_skills, missing_skills, extra_skills, missing_kws = [], [], [], []
    if has_jd:
        matching_skills, missing_skills, extra_skills = find_skill_overlap(
            resume_text, jd_text)
        missing_kws = missing_keywords_from_jd(resume_text, jd_text)

    final_score = compute_final_score(h_score, similarity, has_jd)

    feedback = []
    if contact_issues:
        feedback.append({'type': 'error', 'title': 'Contact Info Issues',
                        'body': ' | '.join(contact_issues)})
    if sections_missing:
        feedback.append({'type': 'warning', 'title': 'Missing Sections',
                         'body': f"Not detected: {', '.join(s.title() for s in sections_missing)}"})
    if quant_ratio < 0.3:
        samples = '; '.join(f'"{u[:55]}..."' for u in unquantified[:2])
        feedback.append({'type': 'error', 'title': 'Add Numbers to Bullets',
                         'body': f'Only {int(quant_ratio*100)}% of bullets have metrics. e.g. {samples}'})
    elif quant_ratio < 0.6:
        feedback.append({'type': 'warning', 'title': 'More Metrics Needed',
                         'body': f'{int(quant_ratio*100)}% of bullets have numbers. Aim for 60%+.'})
    else:
        feedback.append({'type': 'success', 'title': 'Good Use of Numbers',
                         'body': f'{int(quant_ratio*100)}% of your bullets include metrics.'})
    if weak_verbs:
        feedback.append({'type': 'warning', 'title': 'Weak Language Detected',
                         'body': 'Replace: '+' | '.join(f'"{w[:55]}"' for w in weak_verbs[:2])})
    if length_status == 'too_short':
        feedback.append({'type': 'warning', 'title': 'Resume Too Short',
                        'body': f'{word_count} words. Add more project details.'})
    elif length_status == 'too_long':
        feedback.append({'type': 'warning', 'title': 'Resume Too Long',
                        'body': f'{word_count} words. Trim to under 700.'})
    else:
        feedback.append({'type': 'success', 'title': 'Good Length',
                        'body': f'{word_count} words — fits a 1-page resume.'})
    if cliches:
        feedback.append(
            {'type': 'warning', 'title': 'Clichés Found', 'body': ' | '.join(cliches)})
    if has_jd:
        pct = round(similarity * 100)
        if pct >= 60:
            feedback.append({'type': 'success', 'title': f'Strong JD Match ({pct}%)',
                            'body': 'Your resume aligns well with the job description.'})
        elif pct >= 35:
            feedback.append({'type': 'warning', 'title': f'Moderate JD Match ({pct}%)',
                            'body': 'Some alignment but key JD terms are missing from your resume.'})
        else:
            feedback.append({'type': 'error', 'title': f'Weak JD Match ({pct}%)',
                            'body': 'Your resume needs more keywords from the job description.'})

    return {
        'score':           final_score,
        'heuristic_score': h_score,
        'similarity_pct':  round(similarity * 100, 1),
        'has_jd':          has_jd,
        'word_count':      word_count,
        'sections_found':  sections_found,
        'sections_missing': sections_missing,
        'matching_skills': matching_skills,
        'missing_skills':  missing_skills,
        'missing_keywords': missing_kws,
        'feedback':        feedback,
        'breakdown': {
            'Sections':    {'score': min(20, len(sections_found)*4),                                        'max': 20},
            'Contact':     {'score': max(0, 12 - len(contact_issues)*4),                                    'max': 12},
            'Metrics':     {'score': int(quant_ratio * 14),                                                 'max': 14},
            'Verbs':       {'score': max(0, min(8, len(strong_verbs)*2) - min(6, len(weak_verbs)*2)),       'max': 8},
            'Length':      {'score': 6 if length_status == 'good' else 3,                                     'max': 6},
            'JD Match':    {'score': round(similarity*40) if has_jd else 0,                                 'max': 40},
        }
    }


@app.route('/')
def index():
    return render_template('website.html')


@app.route('/analyze', methods=['POST'])
def analyze_route():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['resume']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in {'pdf', 'docx', 'txt'}:
        return jsonify({'error': 'Only PDF, DOCX, or TXT files are supported'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    resume_text = extract_text(filepath, filename)
    os.remove(filepath)

    if not resume_text or len(resume_text.strip()) < 50:
        return jsonify({'error': 'Could not read text from resume. Try DOCX or TXT.'}), 400

    jd_text = request.form.get('jd_text', '').strip()
    return jsonify(analyze(resume_text, jd_text))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
