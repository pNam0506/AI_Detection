from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

from .skill_db import SKILL_DATABASE
from .resume_structure_db import RESUME_STRUCTURE_DATABASE

from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import re  


def home(request):
    result = None
    matched_skills = []
    missing_skills = []
    level = None
    recommendations = []
    jd_keywords = []
    found_sections = []
    missing_sections = []
    structure_score = 0
    color_feedback = None
    color_recommendation = None
    layout_feedback = None

    if request.method == "POST":
        resume_file = request.FILES.get("resume")
        job_desc = request.POST.get("job_desc")

        if resume_file and job_desc:

            # ===== 1️⃣ Extract text from PDF (pdfplumber) =====
            resume_text = ""

            resume_file.seek(0)

            with pdfplumber.open(resume_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        resume_text += extracted + "\n"

            resume_text_lower = resume_text.lower()
            job_desc_lower = job_desc.lower()
            
            layout_feedback = analyze_resume_layout(resume_file)
            
            color_feedback, color_recommendation = analyze_resume_color(resume_file)

            

            # ===== 2️⃣ Resume Structure Analysis =====
            found_sections, missing_sections, structure_score = analyze_resume_structure(resume_text)
            
            

            # ===== 3️⃣ Dynamic Keyword Extraction from JD =====
            jd_keywords = extract_top_keywords(job_desc)

            # ===== 4️⃣ TF-IDF Semantic Similarity =====
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )

            vectors = vectorizer.fit_transform([resume_text, job_desc])
            similarity = cosine_similarity(
                vectors[0:1], vectors[1:2]
            )[0][0]

            tfidf_score = similarity * 100

            # ===== 5️⃣ Skill Matching using Static Skill Database =====
            required_skills = []

            for category, skills in SKILL_DATABASE.items():
                for skill in skills:
                    if skill in job_desc_lower:
                        required_skills.append(skill)

            for skill in required_skills:
                if skill in resume_text_lower:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)

            if len(required_skills) > 0:
                skill_score = (len(matched_skills) / len(required_skills)) * 100
            else:
                skill_score = 0

            # ===== 6️⃣ Weighted Hybrid Model =====
            alpha = 0.4   # TF-IDF weight
            beta = 0.6    # Skill weight

            final_score = (alpha * tfidf_score) + (beta * skill_score)

            result = f"AI Resume Match Score: {final_score:.2f}%"

            # ===== 7️⃣ Candidate Classification =====
            if final_score >= 75:
                level = "Highly Suitable Candidate"
            elif final_score >= 50:
                level = "Moderately Suitable Candidate"
            else:
                level = "Low Match"

            # ===== 8️⃣ Auto Keyword Gap Detection =====
            auto_missing_keywords = []

            for keyword in jd_keywords:
                if keyword not in resume_text_lower:
                    auto_missing_keywords.append(keyword)

            for keyword in auto_missing_keywords:
                recommendations.append(
                    f"The job description emphasizes '{keyword}'. Consider adding relevant experience related to {keyword}."
                )

            # ===== 9️⃣ Smart Recommendations =====
            if missing_skills:
                recommendations.append(
                    "Consider adding or emphasizing the following required skills: "
                    + ", ".join(missing_skills)
                )

            if tfidf_score < 50:
                recommendations.append(
                    "Improve alignment between resume content and job description by tailoring project descriptions."
                )

            if skill_score < 60:
                recommendations.append(
                    "Strengthen technical skill coverage based on the job requirements."
                )

            if 50 <= final_score < 75:
                recommendations.append(
                    "Enhance measurable achievements and quantify project impacts."
                )

            if final_score >= 75:
                recommendations.append(
                    "Your resume is strongly aligned. Minor refinements in clarity and formatting may further improve presentation."
                )

    return render(request, "home.html", {
        "result": result,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "level": level,
        "recommendations": recommendations,
        "jd_keywords": jd_keywords,
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "structure_score": structure_score,
        "color_feedback": color_feedback,
        "color_recommendation": color_recommendation,
        "layout_feedback": layout_feedback
    })


# =========================
# 🔎 Keyword Extraction
# =========================
def extract_top_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=50
    )

    X = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]

    keyword_scores = list(zip(features, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)

    top_keywords = [kw for kw, score in keyword_scores[:top_n]]
    return top_keywords


# =========================
# 📄 Resume Structure Analyzer
# =========================
def analyze_resume_structure(resume_text):
    resume_text_lower = resume_text.lower()

    found_sections = []
    missing_sections = []

    total_weight = 0
    achieved_weight = 0

    for section, data in RESUME_STRUCTURE_DATABASE.items():
        keywords = data["keywords"]
        weight = data["weight"]

        total_weight += weight

        # =====================
        # 📱 ตรวจ Phone แบบพิเศษ
        # =====================
        if section == "Phone":
            phone_pattern = re.search(r'\+?\d[\d\s\-]{8,}', resume_text)
            if phone_pattern:
                found_sections.append(section)
                achieved_weight += weight
            else:
                missing_sections.append(section)

        # =====================
        # 📧 ตรวจ Email แบบพิเศษ
        # =====================
        elif section == "Email":
            if "@" in resume_text:
                found_sections.append(section)
                achieved_weight += weight
            else:
                missing_sections.append(section)

        # =====================
        # 🧠 ตรวจ Section อื่นตาม keyword ปกติ
        # =====================
        else:
            if any(keyword in resume_text_lower for keyword in keywords):
                found_sections.append(section)
                achieved_weight += weight
            else:
                missing_sections.append(section)

    if total_weight > 0:
        structure_score = (achieved_weight / total_weight) * 100
    else:
        structure_score = 0

    return found_sections, missing_sections, structure_score

def analyze_resume_color(pdf_file):
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()

    images = convert_from_bytes(
        pdf_bytes,
        first_page=1,
        last_page=1,
        poppler_path=r"C:\Users\Pinmanee\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    )

    img = images[0]
    img = img.resize((300, 300))

    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)

    avg_color = np.mean(pixels, axis=0)
    r, g, b = avg_color

    brightness = (r + g + b) / 3
    color_variance = np.std(pixels)

    recommendation = ""

    if brightness > 240:
        feedback = "Resume is very bright and minimal."
        recommendation = (
            "Consider adding subtle accent colors such as Navy Blue (#1F3A8A) "
            "or Dark Gray (#374151) to enhance visual hierarchy."
        )

    elif color_variance > 70:
        feedback = "Highly colorful resume detected."
        recommendation = (
            "For corporate or technical roles, consider neutral tones like "
            "Dark Blue (#1E3A8A), Charcoal (#333333), or Slate Gray (#475569)."
        )

    else:
        feedback = "Professional color balance detected."
        recommendation = (
            "Maintain current tone. For refinement, use one primary color "
            "and one accent color to ensure consistency."
        )

    pdf_file.seek(0)

    return feedback, recommendation

# def analyze_resume_layout(pdf_file):
#     pdf_file.seek(0)
#     layout_feedback = ""

#     with pdfplumber.open(pdf_file) as pdf:
#         page = pdf.pages[0]
#         words = page.extract_words()

#         x_positions = [word["x0"] for word in words]

#         if not x_positions:
#             return "Unable to analyze layout."

#         min_x = min(x_positions)
#         max_x = max(x_positions)
#         spread = max_x - min_x

#         # ตรวจว่ามี text กระจายสองฝั่งไหม
#         left_side = [x for x in x_positions if x < (min_x + spread/2)]
#         right_side = [x for x in x_positions if x >= (min_x + spread/2)]

#         if len(left_side) > 0 and len(right_side) > 0:
#             layout_feedback = (
#                 "เรซูเม่ของคุณใช้รูปแบบ 2 คอลัมน์\n"
#                 "✔ ดูทันสมัยและแบ่งข้อมูลชัดเจน\n"
#                 "⚠ แนะนำให้วาง Skills และข้อมูลสำคัญไว้ฝั่งซ้ายหรือด้านบน\n"
#                 "⚠ หลีกเลี่ยงการใส่ข้อมูลสำคัญไว้ในกล่องหรือกราฟิก เพราะระบบ ATS อาจอ่านไม่ครบ"
#             )
#         else:
#             layout_feedback = (
#                 "เรซูเม่ของคุณใช้รูปแบบคอลัมน์เดียว\n"
#                 "✔ อ่านง่าย เป็นมิตรกับระบบ ATS\n"
#                 "💡 แนะนำจัดลำดับให้ Skills และประสบการณ์อยู่ช่วงบนของหน้า"
#             )

#     pdf_file.seek(0)
#     return layout_feedback

def analyze_resume_layout(pdf_file):
    pdf_file.seek(0)

    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()

        if not words:
            return "ไม่สามารถวิเคราะห์รูปแบบการจัดวางได้"

        feedback = []

        # =====================
        # 1️⃣ ตรวจคอลัมน์
        # =====================
        x_positions = [word["x0"] for word in words]
        min_x = min(x_positions)
        max_x = max(x_positions)
        spread = max_x - min_x

        left_side = [x for x in x_positions if x < (min_x + spread/2)]
        right_side = [x for x in x_positions if x >= (min_x + spread/2)]

        if len(left_side) > 0 and len(right_side) > 0:
            feedback.append("📌 ใช้รูปแบบ 2 คอลัมน์ (ดูทันสมัย)")
            feedback.append("⚠ ควรวาง Skills และข้อมูลสำคัญไว้ด้านบนหรือฝั่งซ้าย")
        else:
            feedback.append("📌 ใช้รูปแบบคอลัมน์เดียว (อ่านง่ายและเหมาะกับ ATS)")

        # =====================
        # 2️⃣ ตรวจความแน่นของข้อมูล
        # =====================
        page_area = page.width * page.height
        text_density = len(words) / page_area

        if text_density > 0.0025:
            feedback.append("⚠ เนื้อหาค่อนข้างแน่น ลองเพิ่มระยะห่างหรือแบ่ง bullet ให้สั้นลง")
        else:
            feedback.append("✔ ความหนาแน่นของข้อความอยู่ในระดับอ่านสบาย")

        # =====================
        # 3️⃣ ตรวจขนาดฟอนต์โดยประมาณ
        # =====================
        sizes = [float(word["size"]) for word in words if "size" in word]

        if sizes:
            avg_font_size = sum(sizes) / len(sizes)

            if avg_font_size < 9:
                feedback.append("⚠ ฟอนต์ค่อนข้างเล็ก อาจอ่านยากเมื่อพิมพ์ออกกระดาษ")
            elif avg_font_size > 13:
                feedback.append("⚠ ฟอนต์ค่อนข้างใหญ่ อาจทำให้เนื้อหาดูไม่กระชับ")
            else:
                feedback.append("✔ ขนาดฟอนต์อยู่ในช่วงที่เหมาะสม")
        else:
            feedback.append("ไม่สามารถประเมินขนาดฟอนต์ได้")

        # =====================
        # 4️⃣ ตรวจ spacing แนวตั้ง
        # =====================
        y_positions = sorted([word["top"] for word in words])
        gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]

        if gaps:
            avg_gap = sum(gaps) / len(gaps)

            if avg_gap < 5:
                feedback.append("⚠ ระยะห่างระหว่างบรรทัดค่อนข้างชิด อาจทำให้อ่านยาก")
            else:
                feedback.append("✔ ระยะห่างระหว่างบรรทัดเหมาะสม")

    pdf_file.seek(0)

    return "\n".join(feedback)

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Load model
model_path = os.path.join(settings.BASE_DIR, "body_language.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@csrf_exempt
def analyze_body_language(request):
    if request.method == "POST":

        frame_data = request.FILES.get("frame")
        if not frame_data:
            return JsonResponse({"error": "No frame received"})

        file_bytes = np.frombuffer(frame_data.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ ใช้ Holistic แทน Pose
        with mp.solutions.holistic.Holistic() as holistic:
            results = holistic.process(img_rgb)

        landmarks = []

        # ===== Pose =====
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0] * (33 * 4))

        # ===== Face =====
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0] * (468 * 4))

        # ❌ เอา Left Hand ออก
        # ❌ เอา Right Hand ออก

        print("Feature length:", len(landmarks))

        input_data = np.array(landmarks).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        return JsonResponse({
            "prediction": str(prediction)
        })

    return JsonResponse({"error": "Invalid request"})