import os
import re
from pypdf import PdfReader
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer Elite",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# LANGUAGE PACK
# -----------------------------
LANG = {
    "English": {
        "title": "AI Resume Analyzer Elite",
        "hero": "Upload a PDF resume or paste resume text, compare it with a job description, and get a premium-style hiring analysis.",
        "language": "Language",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "options": "Options",
        "resume_mode": "Resume Input Type",
        "upload_pdf": "Upload PDF Resume",
        "paste_text": "Paste Resume Text",
        "load_demo": "Load Demo CV",
        "premium_access": "Premium Access",
        "premium_code": "Enter premium code",
        "premium_placeholder": "Premium code",
        "premium_unlocked": "Premium unlocked",
        "premium_locked": "Some advanced features are premium only.",
        "features": "Features",
        "pdf_upload": "PDF CV Upload",
        "resume_input": "Resume Input",
        "upload_cv_pdf": "Upload CV as PDF",
        "pdf_success": "PDF text extracted successfully.",
        "pdf_error": "Could not extract readable text from this PDF.",
        "preview_resume": "Preview extracted resume text",
        "paste_resume": "Paste your resume text",
        "paste_resume_placeholder": "Paste the resume content here...",
        "job_description": "Job Description",
        "paste_jd": "Paste the target job description",
        "paste_jd_placeholder": "Paste the job description here...",
        "run_analysis": "Run Elite Analysis",
        "warning_fields": "Please provide both a resume and a job description.",
        "match_score": "Match Score",
        "ats_score": "ATS Score",
        "job_chance": "Job Chance",
        "role_prediction": "Role Prediction",
        "overall_evaluation": "Overall Evaluation",
        "resume_rating": "Resume Rating",
        "recruiter_view": "Recruiter View",
        "premium_only": "Premium Only",
        "unlock_premium_msg": "Unlock Premium to view job chance estimate and recruiter impression.",
        "summary_title": "CV Summary",
        "predicted_role": "Predicted Best-Fit Role",
        "matched_skills": "Matched Skills",
        "skill_gap": "Skill Gap Detection",
        "no_matched": "No matched skills found.",
        "no_missing": "No major missing skills detected.",
        "premium_features": "Premium Features",
        "unlock_premium_to_access": "Unlock premium to access:",
        "download_report": "Download Analysis Report",
        "report_filename": "resume_analysis_report.txt",
        "demo_note": "These results are demo-style estimates based on text similarity and keyword matching. They are not real hiring guarantees.",
        "footer": "AI Resume Analyzer Elite • premium CV screening demo",
        "tip_banner": "Tip: Add a file named banner.png to the same folder if you want a hero image.",
        "strong_match": "Strong Match",
        "moderate_match": "Moderate Match",
        "weak_match": "Weak Match",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "strong_candidate": "Strong candidate",
        "potential_candidate": "Potential candidate",
        "needs_improvement": "Needs improvement",
        "role_undetermined": "General / Undetermined",
        "job_chance_text_high": "Your resume looks competitive for this role.",
        "job_chance_text_medium": "You may have a reasonable chance, but your CV could be improved.",
        "job_chance_text_low": "Your current CV may struggle against this job description.",
        "recruiter_text_strong": "This resume gives a positive first impression for screening.",
        "recruiter_text_medium": "The profile shows promise, but there are areas to improve.",
        "recruiter_text_low": "The resume may not stand out strongly in an initial recruiter review.",
    },
    "Türkçe": {
        "title": "Yapay Zekâ CV Analizörü Elite",
        "hero": "PDF CV yükleyin veya CV metnini yapıştırın, iş ilanı ile karşılaştırın ve premium görünümlü bir işe alım analizi alın.",
        "language": "Dil",
        "theme": "Tema",
        "light": "Açık",
        "dark": "Koyu",
        "options": "Seçenekler",
        "resume_mode": "CV Giriş Türü",
        "upload_pdf": "PDF CV Yükle",
        "paste_text": "CV Metni Yapıştır",
        "load_demo": "Demo CV Yükle",
        "premium_access": "Premium Erişim",
        "premium_code": "Premium kodunu gir",
        "premium_placeholder": "Premium kodu",
        "premium_unlocked": "Premium açıldı",
        "premium_locked": "Bazı gelişmiş özellikler sadece premium.",
        "features": "Özellikler",
        "pdf_upload": "PDF CV Yükleme",
        "resume_input": "CV Girişi",
        "upload_cv_pdf": "CV'yi PDF olarak yükle",
        "pdf_success": "PDF metni başarıyla çıkarıldı.",
        "pdf_error": "Bu PDF'ten okunabilir metin çıkarılamadı.",
        "preview_resume": "Çıkarılan CV metnini önizle",
        "paste_resume": "CV metnini yapıştır",
        "paste_resume_placeholder": "CV içeriğini buraya yapıştır...",
        "job_description": "İş İlanı",
        "paste_jd": "Hedef iş ilanını yapıştır",
        "paste_jd_placeholder": "İş ilanını buraya yapıştır...",
        "run_analysis": "Elite Analizi Çalıştır",
        "warning_fields": "Lütfen hem CV hem iş ilanı gir.",
        "match_score": "Uyum Skoru",
        "ats_score": "ATS Skoru",
        "job_chance": "İşe Girme İhtimali",
        "role_prediction": "Rol Tahmini",
        "overall_evaluation": "Genel Değerlendirme",
        "resume_rating": "CV Değerlendirmesi",
        "recruiter_view": "Recruiter Görüşü",
        "premium_only": "Sadece Premium",
        "unlock_premium_msg": "İşe girme ihtimali ve recruiter görüşünü görmek için Premium'u aç.",
        "summary_title": "CV Özeti",
        "predicted_role": "Tahmini En Uygun Rol",
        "matched_skills": "Eşleşen Yetkinlikler",
        "skill_gap": "Eksik Yetkinlik Tespiti",
        "no_matched": "Eşleşen yetkinlik bulunamadı.",
        "no_missing": "Belirgin bir eksik yetkinlik bulunmadı.",
        "premium_features": "Premium Özellikler",
        "unlock_premium_to_access": "Şunlara erişmek için premium aç:",
        "download_report": "Analiz Raporunu İndir",
        "report_filename": "cv_analiz_raporu.txt",
        "demo_note": "Bu sonuçlar metin benzerliği ve anahtar kelime eşleşmesine dayalı demo tahminlerdir. Gerçek işe alım garantisi vermez.",
        "footer": "Yapay Zekâ CV Analizörü Elite • premium CV tarama demosu",
        "tip_banner": "İstersen aynı klasöre banner.png ekleyerek üst görsel kullanabilirsin.",
        "strong_match": "Güçlü Uyum",
        "moderate_match": "Orta Uyum",
        "weak_match": "Zayıf Uyum",
        "high": "Yüksek",
        "medium": "Orta",
        "low": "Düşük",
        "strong_candidate": "Güçlü aday",
        "potential_candidate": "Potansiyelli aday",
        "needs_improvement": "Geliştirilmeli",
        "role_undetermined": "Genel / Belirsiz",
        "job_chance_text_high": "CV'n bu rol için rekabetçi görünüyor.",
        "job_chance_text_medium": "Makul bir şansın olabilir ama CV daha güçlü hale getirilebilir.",
        "job_chance_text_low": "Mevcut CV bu iş ilanına karşı zayıf kalabilir.",
        "recruiter_text_strong": "Bu CV ilk elemede olumlu bir izlenim veriyor.",
        "recruiter_text_medium": "Profil umut veriyor ancak geliştirilecek alanlar var.",
        "recruiter_text_low": "Bu CV ilk recruiter değerlendirmesinde yeterince öne çıkmayabilir.",
    }
}

# -----------------------------
# DATA
# -----------------------------
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with",
    "to", "of", "in", "on", "for", "from", "by", "is", "are",
    "was", "were", "be", "been", "being", "at", "as", "it",
    "this", "that", "these", "those", "your", "you", "our",
    "we", "their", "them", "they", "will", "can", "could",
    "should", "would", "may", "might", "into", "about"
}

COMMON_SKILLS = [
    "python", "java", "c++", "c", "sql", "machine learning", "deep learning",
    "data analysis", "data science", "excel", "power bi", "tableau",
    "communication", "teamwork", "leadership", "problem solving",
    "project management", "nlp", "tensorflow", "pytorch", "research",
    "statistics", "api", "javascript", "html", "css", "git", "streamlit",
    "marketing", "sales", "customer service", "public speaking", "writing",
    "critical thinking", "time management", "collaboration", "automation",
    "clinical", "rehabilitation", "physiotherapy", "healthcare", "analysis",
    "presentation", "documentation", "customer support", "crm",
    "content writing", "social media", "teaching", "training"
]

ROLE_KEYWORDS = {
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "statistics", "python"],
    "Machine Learning / AI": ["machine learning", "deep learning", "python", "tensorflow", "pytorch", "nlp"],
    "Software Developer": ["python", "java", "c++", "javascript", "html", "css", "api", "git"],
    "Marketing / Content": ["marketing", "writing", "content writing", "social media", "communication"],
    "Project / Operations": ["project management", "leadership", "communication", "collaboration", "time management"],
    "Healthcare / Rehabilitation": ["healthcare", "clinical", "rehabilitation", "physiotherapy", "communication"],
    "Research / Academic": ["research", "analysis", "writing", "statistics", "presentation", "documentation"]
}

DEMO_CV_EN = """
Alex Morgan
Email: alex@example.com

Profile
Data-driven analyst with experience in Python, SQL, Excel, Power BI, research, and dashboard reporting.
Strong communication, teamwork, and problem solving skills.

Experience
- Built dashboards in Power BI for weekly KPI tracking
- Analyzed operational data using Python and SQL
- Collaborated with teams to improve reporting accuracy
- Presented findings to stakeholders

Skills
Python, SQL, Excel, Power BI, Data Analysis, Research, Communication, Teamwork, Statistics
"""

DEMO_CV_TR = """
Aybike Yılmaz
E-posta: aybike@example.com

Profil
Python, SQL, Excel, Power BI, araştırma ve raporlama deneyimine sahip veri odaklı analist.
Güçlü iletişim, takım çalışması ve problem çözme becerileri.

Deneyim
- Haftalık KPI takibi için Power BI panoları hazırladım
- Python ve SQL ile operasyonel veri analizi yaptım
- Ekiplerle birlikte raporlama doğruluğunu geliştirdim
- Bulguları paydaşlara sundum

Yetkinlikler
Python, SQL, Excel, Power BI, Veri Analizi, Araştırma, İletişim, Takım Çalışması, İstatistik
"""

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_css(theme: str) -> str:
    if theme == "dark":
        return """
        <style>
        .main { background: linear-gradient(135deg, #0b1220 0%, #111827 55%, #1f2937 100%); color: #f3f4f6; }
        .block-container { max-width: 1200px; padding-top: 1.6rem; padding-bottom: 2rem; }
        .hero-box { background: linear-gradient(135deg, #111827 0%, #1d4ed8 45%, #7c3aed 100%); padding: 34px; border-radius: 28px; color: white; box-shadow: 0 20px 45px rgba(29,78,216,0.18); margin-bottom: 22px; }
        .hero-title { font-size: 42px; font-weight: 900; margin-bottom: 8px; }
        .card { background: rgba(17,24,39,0.92); border: 1px solid rgba(55,65,81,0.9); padding: 22px; border-radius: 24px; box-shadow: 0 10px 26px rgba(0,0,0,0.25); margin-bottom: 18px; color: #f3f4f6; }
        .section-title { font-size: 18px; font-weight: 800; color: #f9fafb; margin-bottom: 10px; }
        .metric-box { background: linear-gradient(135deg, #111827 0%, #1f2937 100%); border: 1px solid #374151; padding: 18px; border-radius: 22px; text-align: center; box-shadow: 0 6px 16px rgba(0,0,0,0.2); }
        .metric-label { font-size: 14px; color: #cbd5e1; font-weight: 700; }
        .metric-value { font-size: 30px; font-weight: 900; color: #f9fafb; margin-top: 4px; }
        .big-score { font-size: 52px; font-weight: 900; color: #c4b5fd; line-height: 1; }
        .tag { display: inline-block; padding: 8px 12px; border-radius: 999px; margin: 4px 6px 4px 0; font-size: 14px; font-weight: 700; }
        .good { background-color: #14532d; color: #bbf7d0; }
        .warn { background-color: #78350f; color: #fde68a; }
        .bad { background-color: #7f1d1d; color: #fecaca; }
        .pill { display: inline-block; padding: 10px 14px; border-radius: 999px; font-weight: 800; font-size: 14px; margin-top: 8px; margin-right: 8px; }
        .pill-good { background: #14532d; color: #bbf7d0; }
        .pill-mid { background: #78350f; color: #fde68a; }
        .pill-bad { background: #7f1d1d; color: #fecaca; }
        .premium-lock {
            position: relative; overflow: hidden; min-height: 180px; border-radius: 24px;
            background: rgba(17,24,39,0.92); border: 1px solid rgba(55,65,81,0.9); padding: 22px; margin-bottom: 18px;
        }
        .premium-blur { filter: blur(8px); opacity: 0.35; pointer-events: none; user-select: none; }
        .premium-overlay {
            position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            flex-direction: column; font-weight: 800; color: #f9fafb; backdrop-filter: blur(2px);
        }
        .footer-note { text-align: center; color: #9ca3af; font-size: 14px; margin-top: 24px; }
        </style>
        """
    return """
    <style>
    .main { background: linear-gradient(135deg, #f8fbff 0%, #eef2ff 45%, #f8fafc 100%); }
    .block-container { max-width: 1200px; padding-top: 1.6rem; padding-bottom: 2rem; }
    .hero-box { background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #7c3aed 100%); padding: 34px; border-radius: 28px; color: white; box-shadow: 0 20px 45px rgba(29,78,216,0.18); margin-bottom: 22px; }
    .hero-title { font-size: 42px; font-weight: 900; margin-bottom: 8px; }
    .card { background: rgba(255,255,255,0.93); border: 1px solid rgba(226,232,240,0.9); padding: 22px; border-radius: 24px; box-shadow: 0 10px 26px rgba(15,23,42,0.06); margin-bottom: 18px; }
    .section-title { font-size: 18px; font-weight: 800; color: #111827; margin-bottom: 10px; }
    .metric-box { background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border: 1px solid #e2e8f0; padding: 18px; border-radius: 22px; text-align: center; box-shadow: 0 6px 16px rgba(15,23,42,0.05); }
    .metric-label { font-size: 14px; color: #64748b; font-weight: 700; }
    .metric-value { font-size: 30px; font-weight: 900; color: #0f172a; margin-top: 4px; }
    .big-score { font-size: 52px; font-weight: 900; color: #312e81; line-height: 1; }
    .tag { display: inline-block; padding: 8px 12px; border-radius: 999px; margin: 4px 6px 4px 0; font-size: 14px; font-weight: 700; }
    .good { background-color: #dcfce7; color: #166534; }
    .warn { background-color: #fef3c7; color: #92400e; }
    .bad { background-color: #fee2e2; color: #991b1b; }
    .pill { display: inline-block; padding: 10px 14px; border-radius: 999px; font-weight: 800; font-size: 14px; margin-top: 8px; margin-right: 8px; }
    .pill-good { background: #dcfce7; color: #166534; }
    .pill-mid { background: #fef3c7; color: #92400e; }
    .pill-bad { background: #fee2e2; color: #991b1b; }
    .premium-lock {
        position: relative; overflow: hidden; min-height: 180px; border-radius: 24px;
        background: rgba(255,255,255,0.93); border: 1px solid rgba(226,232,240,0.9); padding: 22px; margin-bottom: 18px;
    }
    .premium-blur { filter: blur(8px); opacity: 0.35; pointer-events: none; user-select: none; }
    .premium-overlay {
        position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
        flex-direction: column; font-weight: 800; color: #111827; backdrop-filter: blur(2px);
    }
    .footer-note { text-align: center; color: #6b7280; font-size: 14px; margin-top: 24px; }
    </style>
    """

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        return ""
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

def extract_skills(text: str, skills_list: list[str]) -> list[str]:
    text_lower = text.lower()
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    return sorted(list(set(found_skills)))

def calculate_similarity(cv_text: str, jd_text: str) -> float:
    if not cv_text.strip() or not jd_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cv_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(similarity) * 100, 2)

def classify_match(score: float, t: dict) -> tuple[str, str]:
    if score >= 78:
        return t["strong_match"], "pill-good"
    if score >= 55:
        return t["moderate_match"], "pill-mid"
    return t["weak_match"], "pill-bad"

def estimate_job_chance(score: float, matched_skills: list[str], missing_skills: list[str], t: dict) -> tuple[int, str, str, str]:
    bonus = min(len(matched_skills) * 2, 12)
    penalty = min(len(missing_skills) * 1.5, 15)
    chance = score + bonus - penalty
    chance = max(5, min(95, round(chance)))

    if chance >= 75:
        return chance, t["high"], "pill-good", t["job_chance_text_high"]
    if chance >= 50:
        return chance, t["medium"], "pill-mid", t["job_chance_text_medium"]
    return chance, t["low"], "pill-bad", t["job_chance_text_low"]

def calculate_ats_score(score: float, matched_skills: list[str], missing_skills: list[str], cv_text: str) -> int:
    keyword_score = min(len(matched_skills) * 6, 40)
    balance_score = max(0, 30 - len(missing_skills) * 3)
    content_score = 15 if len(cv_text.split()) >= 120 else 8
    similarity_part = min(score * 0.15, 15)
    ats = round(keyword_score + balance_score + content_score + similarity_part)
    return max(0, min(100, ats))

def recruiter_impression(score: float, ats_score: int, missing_skills: list[str], t: dict) -> tuple[str, str, str]:
    avg = (score + ats_score) / 2
    if avg >= 75 and len(missing_skills) <= 4:
        return t["strong_candidate"], "pill-good", t["recruiter_text_strong"]
    if avg >= 55:
        return t["potential_candidate"], "pill-mid", t["recruiter_text_medium"]
    return t["needs_improvement"], "pill-bad", t["recruiter_text_low"]

def predict_role(cv_text: str, t: dict) -> str:
    text = cv_text.lower()
    scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        role_score = 0
        for keyword in keywords:
            if keyword.lower() in text:
                role_score += 1
        scores[role] = role_score
    best_role = max(scores, key=scores.get)
    if scores[best_role] == 0:
        return t["role_undetermined"]
    return best_role

def generate_cv_summary(predicted_role: str, matched_skills: list[str], score: float, lang_choice: str) -> str:
    top_skills = ", ".join(matched_skills[:5]) if matched_skills else (
        "general transferable skills" if lang_choice == "English" else "genel aktarılabilir beceriler"
    )
    if lang_choice == "English":
        if score >= 75:
            return f"This resume appears well aligned with {predicted_role} opportunities and highlights relevant capabilities such as {top_skills}."
        if score >= 55:
            return f"This resume shows partial fit for {predicted_role} roles, with some relevant strengths including {top_skills}, but stronger targeting is needed."
        return f"This resume currently presents a weaker match for the selected role. It may benefit from clearer positioning toward {predicted_role} work and stronger emphasis on role-specific skills."
    else:
        if score >= 75:
            return f"Bu CV, {predicted_role} fırsatlarıyla oldukça uyumlu görünüyor ve {top_skills} gibi ilgili yetkinlikleri öne çıkarıyor."
        if score >= 55:
            return f"Bu CV, {predicted_role} rolleri için kısmi uyum gösteriyor; {top_skills} gibi bazı güçlü yönler var ancak daha iyi hedefleme gerekiyor."
        return f"Bu CV şu anda seçilen rol için daha zayıf bir uyum gösteriyor. {predicted_role} yönünde daha net konumlandırma ve role özgü becerilerin daha güçlü vurgulanması faydalı olabilir."

def build_report(t: dict, lang_choice: str, score: float, ats: int, predicted_role: str, summary: str,
                 matched: list[str], missing: list[str], is_premium: bool, chance: int, chance_label: str,
                 recruiter_label: str) -> bytes:
    lines = []
    if lang_choice == "English":
        lines.extend([
            "AI Resume Analyzer Elite Report",
            "--------------------------------",
            f"Match Score: {score}%",
            f"ATS Score: {ats}%",
            f"Role Prediction: {predicted_role}",
            f"CV Summary: {summary}",
            f"Matched Skills: {', '.join(matched) if matched else 'None'}",
        ])
        if is_premium:
            lines.append(f"Missing Skills: {', '.join(missing) if missing else 'None'}")
            lines.append(f"Job Chance: {chance}% ({chance_label})")
            lines.append(f"Recruiter Impression: {recruiter_label}")
        else:
            lines.append("Missing Skills: Premium only")
            lines.append("Job Chance: Premium only")
            lines.append("Recruiter Impression: Premium only")
    else:
        lines.extend([
            "Yapay Zekâ CV Analizörü Elite Raporu",
            "-----------------------------------",
            f"Uyum Skoru: {score}%",
            f"ATS Skoru: {ats}%",
            f"Rol Tahmini: {predicted_role}",
            f"CV Özeti: {summary}",
            f"Eşleşen Yetkinlikler: {', '.join(matched) if matched else 'Yok'}",
        ])
        if is_premium:
            lines.append(f"Eksik Yetkinlikler: {', '.join(missing) if missing else 'Yok'}")
            lines.append(f"İşe Girme İhtimali: {chance}% ({chance_label})")
            lines.append(f"Recruiter Görüşü: {recruiter_label}")
        else:
            lines.append("Eksik Yetkinlikler: Sadece premium")
            lines.append("İşe Girme İhtimali: Sadece premium")
            lines.append("Recruiter Görüşü: Sadece premium")
    return "\n".join(lines).encode("utf-8")

# -----------------------------
# SESSION STATE
# -----------------------------
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False
if "cv_text_manual" not in st.session_state:
    st.session_state.cv_text_manual = ""

# -----------------------------
# TOP CONTROLS
# -----------------------------
with st.sidebar:
    lang_choice = st.selectbox("Language / Dil", ["English", "Türkçe"])
    t = LANG[lang_choice]
    theme_choice = st.radio(t["theme"], [t["light"], t["dark"]], index=0)
    theme_mode = "dark" if theme_choice == t["dark"] else "light"

st.markdown(get_css(theme_mode), unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
banner_path = "banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">{t['title']}</div>
        <div>{t['hero']}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
with st.sidebar:
    st.header(t["options"])
    mode = st.radio(t["resume_mode"], [t["upload_pdf"], t["paste_text"]])

    if st.button(f"🧪 {t['load_demo']}", use_container_width=True):
        st.session_state.demo_loaded = True
        st.session_state.cv_text_manual = DEMO_CV_EN if lang_choice == "English" else DEMO_CV_TR

    st.markdown("---")
    st.markdown(f"### {t['premium_access']}")
    premium_unlocked = st.text_input(
        t["premium_code"],
        type="password",
        placeholder=t["premium_placeholder"]
    )
    is_premium = premium_unlocked == "elite123"

    if is_premium:
        st.success(t["premium_unlocked"])
    else:
        st.info(t["premium_locked"])

    st.markdown("---")
    st.markdown(f"### {t['features']}")
    st.write(f"✅ {t['pdf_upload']}")
    st.write(f"✅ {t['match_score']}")
    st.write(f"✅ {t['ats_score']}")
    st.write(f"✅ {t['role_prediction']}")
    st.write(f"✅ {t['summary_title']}")
    st.write(f"🔒 {t['skill_gap']}")
    st.write(f"🔒 {t['job_chance']}")
    st.write(f"🔒 {t['recruiter_view']}")
    st.markdown("---")
    st.caption(t["tip_banner"])

# -----------------------------
# INPUTS
# -----------------------------
left_col, right_col = st.columns(2)
cv_text = ""

with left_col:
    st.markdown(f'<div class="card"><div class="section-title">👤 {t["resume_input"]}</div>', unsafe_allow_html=True)

    if mode == t["upload_pdf"]:
        uploaded_pdf = st.file_uploader(t["upload_cv_pdf"], type=["pdf"])
        if uploaded_pdf is not None:
            cv_text = extract_text_from_pdf(uploaded_pdf)
            if cv_text.strip():
                st.success(t["pdf_success"])
                with st.expander(t["preview_resume"]):
                    st.write(cv_text[:3000])
            else:
                st.error(t["pdf_error"])
        elif st.session_state.demo_loaded:
            cv_text = st.session_state.cv_text_manual
            st.info(t["load_demo"])
            st.text_area(
                t["paste_resume"],
                value=cv_text,
                height=350,
                key="demo_preview_pdf_mode"
            )
    else:
        default_text = st.session_state.cv_text_manual
        cv_text = st.text_area(
            t["paste_resume"],
            value=default_text,
            height=350,
            placeholder=t["paste_resume_placeholder"],
            key="manual_resume_input"
        )
        st.session_state.cv_text_manual = cv_text

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(f'<div class="card"><div class="section-title">💼 {t["job_description"]}</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        t["paste_jd"],
        height=350,
        placeholder=t["paste_jd_placeholder"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button(f"✨ {t['run_analysis']}", use_container_width=True)

# -----------------------------
# ANALYSIS
# -----------------------------
if analyze:
    if not cv_text.strip() or not jd_text.strip():
        st.warning(t["warning_fields"])
    else:
        cleaned_cv = clean_text(cv_text)
        cleaned_jd = clean_text(jd_text)

        similarity_score = calculate_similarity(cleaned_cv, cleaned_jd)
        cv_skills = extract_skills(cv_text, COMMON_SKILLS)
        jd_skills = extract_skills(jd_text, COMMON_SKILLS)
        matched_skills = sorted(list(set(cv_skills) & set(jd_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(cv_skills)))

        match_label, match_class = classify_match(similarity_score, t)
        ats_score = calculate_ats_score(similarity_score, matched_skills, missing_skills, cv_text)
        job_chance, chance_label, chance_class, chance_text = estimate_job_chance(similarity_score, matched_skills, missing_skills, t)
        recruiter_label, recruiter_class, recruiter_text = recruiter_impression(similarity_score, ats_score, missing_skills, t)
        predicted_role = predict_role(cv_text, t)
        summary_text = generate_cv_summary(predicted_role, matched_skills, similarity_score, lang_choice)

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{t["match_score"]}</div>
                <div class="metric-value">{similarity_score}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{t["ats_score"]}</div>
                <div class="metric-value">{ats_score}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            if is_premium:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">{t["job_chance"]}</div>
                    <div class="metric-value">{job_chance}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">{t["job_chance"]}</div>
                    <div class="metric-value">🔒 {t["premium_only"]}</div>
                </div>
                """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{t["role_prediction"]}</div>
                <div class="metric-value" style="font-size:18px;">{predicted_role}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="section-title">📊 {t["overall_evaluation"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-score">{similarity_score}%</div>', unsafe_allow_html=True)
        st.progress(min(int(similarity_score), 100))
        st.markdown(f'<span class="pill {match_class}">{match_label}</span>', unsafe_allow_html=True)

        if is_premium:
            st.markdown(f'<span class="pill {chance_class}">{t["job_chance"]}: {chance_label}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="pill {recruiter_class}">{t["recruiter_view"]}: {recruiter_label}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill pill-mid">{t["job_chance"]}: {t["premium_only"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="pill pill-mid">{t["recruiter_view"]}: {t["premium_only"]}</span>', unsafe_allow_html=True)

        st.write("")
        st.write(f"**{t['resume_rating']}:** {match_label}")
        if is_premium:
            st.write(chance_text)
            st.write(recruiter_text)
        else:
            st.write(t["unlock_premium_msg"])

        st.caption(t["demo_note"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="section-title">🧠 {t["summary_title"]}</div>', unsafe_allow_html=True)
        st.write(summary_text)
        st.markdown(f"**{t['predicted_role']}:** {predicted_role}")
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f'<div class="card"><div class="section-title">✅ {t["matched_skills"]}</div>', unsafe_allow_html=True)
            if matched_skills:
                for skill in matched_skills:
                    st.markdown(f'<span class="tag good">{skill}</span>', unsafe_allow_html=True)
            else:
                st.info(t["no_matched"])
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if is_premium:
                st.markdown(f'<div class="card"><div class="section-title">⚠️ {t["skill_gap"]}</div>', unsafe_allow_html=True)
                if missing_skills:
                    for skill in missing_skills:
                        st.markdown(f'<span class="tag bad">{skill}</span>', unsafe_allow_html=True)
                else:
                    st.success(t["no_missing"])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                fake_html = f"""
                <div class="premium-lock">
                    <div class="premium-blur">
                        <div class="section-title">⚠️ {t["skill_gap"]}</div>
                        <span class="tag bad">python</span>
                        <span class="tag bad">sql</span>
                        <span class="tag bad">leadership</span>
                        <span class="tag bad">project management</span>
                    </div>
                    <div class="premium-overlay">
                        <div style="font-size:32px;">🔒</div>
                        <div>{t["premium_only"]}</div>
                    </div>
                </div>
                """
                st.markdown(fake_html, unsafe_allow_html=True)

        if not is_premium:
            st.markdown(f"""
            <div class="card">
                <div class="section-title">🔒 {t["premium_features"]}</div>
                <p>{t["unlock_premium_to_access"]}</p>
                <ul>
                    <li>{t["skill_gap"]}</li>
                    <li>{t["job_chance"]}</li>
                    <li>{t["recruiter_view"]}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        report_bytes = build_report(
            t=t,
            lang_choice=lang_choice,
            score=similarity_score,
            ats=ats_score,
            predicted_role=predicted_role,
            summary=summary_text,
            matched=matched_skills,
            missing=missing_skills,
            is_premium=is_premium,
            chance=job_chance,
            chance_label=chance_label,
            recruiter_label=recruiter_label
        )

        st.download_button(
            label=f"📄 {t['download_report']}",
            data=report_bytes,
            file_name=t["report_filename"],
            mime="text/plain",
            use_container_width=True
        )

st.markdown(f'<div class="footer-note">{t["footer"]}</div>', unsafe_allow_html=True)
