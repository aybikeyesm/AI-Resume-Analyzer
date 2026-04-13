import streamlit as st
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import re
# Basit stopwords listesi (NLTK yerine)
stop_words = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with",
    "to", "of", "in", "on", "for", "from", "by", "is", "are",
    "was", "were", "be", "been", "being", "at", "as", "it",
    "this", "that", "these", "those"
}

COMMON_SKILLS = [
    "python", "java", "c++", "sql", "machine learning", "deep learning",
    "data analysis", "excel", "power bi", "tableau", "communication",
    "teamwork", "leadership", "problem solving", "project management",
    "nlp", "tensorflow", "pytorch", "research", "statistics", "api",
    "javascript", "html", "css", "git", "streamlit"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def extract_skills(text, skills_list):
    text = text.lower()
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill)
    return sorted(list(set(found_skills)))

def calculate_similarity(cv_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cv_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def generate_feedback(score, matched_skills, missing_skills):
    if score >= 75:
        level = "Strong Match"
        comment = "Your CV appears well aligned with the job description."
    elif score >= 50:
        level = "Moderate Match"
        comment = "Your CV matches the role partially, but there is room for improvement."
    else:
        level = "Low Match"
        comment = "Your CV does not strongly match the job description yet."

    feedback = {
        "level": level,
        "comment": comment,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
    return feedback

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("AI Resume Analyzer")
st.write("Paste a resume and a job description to analyze compatibility.")

cv_text = st.text_area("Paste Resume Text", height=250)
jd_text = st.text_area("Paste Job Description", height=250)

if st.button("Analyze Resume"):
    if cv_text.strip() == "" or jd_text.strip() == "":
        st.warning("Please fill in both resume and job description.")
    else:
        cleaned_cv = clean_text(cv_text)
        cleaned_jd = clean_text(jd_text)

        similarity_score = calculate_similarity(cleaned_cv, cleaned_jd)

        cv_skills = extract_skills(cv_text, COMMON_SKILLS)
        jd_skills = extract_skills(jd_text, COMMON_SKILLS)

        matched_skills = sorted(list(set(cv_skills) & set(jd_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(cv_skills)))

        feedback = generate_feedback(similarity_score, matched_skills, missing_skills)

        st.subheader("Analysis Result")
        st.metric("Match Score", f"{similarity_score}%")
        st.write(f"**Match Level:** {feedback['level']}")
        st.write(f"**Comment:** {feedback['comment']}")

        st.subheader("Matched Skills")
        if matched_skills:
            st.success(", ".join(matched_skills))
        else:
            st.info("No matched skills found.")

        st.subheader("Missing Skills")
        if missing_skills:
            st.error(", ".join(missing_skills))
        else:
            st.success("No major missing skills detected.")

        st.subheader("Tips to Improve the Resume")
        if missing_skills:
            for skill in missing_skills:
                st.write(f"- Consider adding or highlighting **{skill}** if you have experience in it.")
        else:
            st.write("- Your resume already covers most of the important skills in this job description.")
