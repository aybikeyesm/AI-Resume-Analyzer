import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Basit stopwords
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

def generate_feedback(score):
    if score >= 75:
        return "Strong Match", "Your resume is highly aligned with the job description."
    elif score >= 50:
        return "Moderate Match", "Your resume matches the job partially, but could be improved."
    else:
        return "Low Match", "Your resume does not yet strongly match the job description."

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero-box {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 32px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 10px 30px rgba(79,70,229,0.25);
    margin-bottom: 24px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 18px;
}
.small-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #1f2937;
}
.big-score {
    font-size: 42px;
    font-weight: 800;
    color: #4f46e5;
}
.tag {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    margin: 4px 6px 4px 0;
    font-size: 14px;
    font-weight: 600;
}
.good {
    background-color: #dcfce7;
    color: #166534;
}
.bad {
    background-color: #fee2e2;
    color: #991b1b;
}
.footer-note {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom:8px;">📄 AI Resume Analyzer</h1>
    <p style="font-size:18px; margin-bottom:0;">
        Compare a resume with a job description, detect matching skills, identify gaps,
        and get a clean compatibility score.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Resume")
    cv_text = st.text_area(
        "Paste your resume text",
        height=320,
        placeholder="Paste the resume content here..."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💼 Job Description")
    jd_text = st.text_area(
        "Paste the job description",
        height=320,
        placeholder="Paste the job description here..."
    )
    st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("✨ Analyze Resume", use_container_width=True)

if analyze:
    if cv_text.strip() == "" or jd_text.strip() == "":
        st.warning("Please fill in both fields.")
    else:
        cleaned_cv = clean_text(cv_text)
        cleaned_jd = clean_text(jd_text)

        similarity_score = calculate_similarity(cleaned_cv, cleaned_jd)
        level, comment = generate_feedback(similarity_score)

        cv_skills = extract_skills(cv_text, COMMON_SKILLS)
        jd_skills = extract_skills(jd_text, COMMON_SKILLS)

        matched_skills = sorted(list(set(cv_skills) & set(jd_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(cv_skills)))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">📊 Compatibility Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-score">{similarity_score}%</div>', unsafe_allow_html=True)
        st.progress(min(int(similarity_score), 100))
        st.write(f"**Match Level:** {level}")
        st.write(f"**Comment:** {comment}")
        st.markdown('</div>', unsafe_allow_html=True)

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="small-title">✅ Matched Skills</div>', unsafe_allow_html=True)
            if matched_skills:
                for skill in matched_skills:
                    st.markdown(f'<span class="tag good">{skill}</span>', unsafe_allow_html=True)
            else:
                st.info("No matched skills found.")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="small-title">⚠️ Missing Skills</div>', unsafe_allow_html=True)
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f'<span class="tag bad">{skill}</span>', unsafe_allow_html=True)
            else:
                st.success("No major missing skills detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">💡 Improvement Tips</div>', unsafe_allow_html=True)
        if missing_skills:
            for skill in missing_skills:
                st.write(f"- Highlight **{skill}** if you already have experience with it.")
        else:
            st.write("- Your resume already covers most of the important requirements.")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Designed for cleaner, smarter CV screening ✨</div>',
    unsafe_allow_html=True
)
