from flask import Flask, render_template, request
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Extract text from PDF
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# Skills list
skills = [
    "python", "java", "c++", "machine learning",
    "data analysis", "html", "css", "javascript"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_skills = []
    missing_skills = []
    match_score = 0
    suggestion = ""

    if request.method == 'POST':
        file = request.files['resume']
        job_desc = request.form.get('job_desc', '')

        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract resume text
        resume_text = extract_text(filepath)

        # Extract skills from resume
        for skill in skills:
            if skill in resume_text:
                extracted_skills.append(skill)

        # Missing skills
        missing_skills = [skill for skill in skills if skill not in extracted_skills]

        # Match Score Calculation (HYBRID)
        if job_desc.strip() != "":
            job_desc_lower = job_desc.lower()

            # Skill-based matching
            job_skills = [skill for skill in skills if skill in job_desc_lower]

            if len(job_skills) > 0:
                matched_skills = [skill for skill in extracted_skills if skill in job_skills]
                skill_score = (len(matched_skills) / len(job_skills)) * 100
            else:
                skill_score = 0

            # Text similarity
            documents = [resume_text, job_desc_lower]
            tfidf = TfidfVectorizer().fit_transform(documents)
            text_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

            # Final score
            match_score = round((0.7 * skill_score) + (0.3 * text_score), 2)

        # Suggestions
        if match_score < 40:
            suggestion = "Your resume needs significant improvement. Add more relevant skills."
        elif match_score < 70:
            suggestion = "Good resume, but you can improve by adding missing skills."
        else:
            suggestion = "Excellent! Your resume matches well."

    return render_template(
        'index.html',
        skills=extracted_skills,
        missing_skills=missing_skills,
        score=match_score,
        suggestion=suggestion
    )

if __name__ == '__main__':
    app.run(debug=True)