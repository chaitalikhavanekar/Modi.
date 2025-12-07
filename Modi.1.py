import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# If needed, set tesseract path (Windows example)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- IMAGE PREPROCESSING ----------

def preprocess_document(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to cleaned, binarized OpenCV image."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


# ---------- OCR ----------

def run_ocr(image: np.ndarray, lang: str = "mar+hin+eng") -> str:
    pil_img = Image.fromarray(image)
    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
    return text


# ---------- DOC TYPE DETECTION ----------

def detect_doc_type(text: str) -> dict:
    t = text.replace("\n", " ")

    land_keywords = ["गाव नमुना", "७/१२", "7/12", "गट क्र", "गट क्रमांक", "सर्वे", "क्षेत्रफळ"]
    budget_keywords = ["बजेट", "आवक-जावक", "आवक जावक", "खर्च"]
    court_keywords = ["न्यायालय", "कोर्ट", "अदालती", "फौजदारी", "दिवाणी"]

    doc_type = "unknown"
    confidence = 0.3

    if any(k in t for k in land_keywords):
        doc_type = "land_record"
        confidence = 0.9
    elif any(k in t for k in budget_keywords):
        doc_type = "budget_or_accounts"
        confidence = 0.85
    elif any(k in t for k in court_keywords):
        doc_type = "court_or_legal"
        confidence = 0.8

    return {"type": doc_type, "confidence": confidence}


# ---------- NAME SAFETY LAYER ----------

def extract_possible_names(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    honorifics = ["श्री", "सौ.", "सौ", "कु."]
    common_surnames = ["पाटील", "जाधव", "देशमुख", "शिंदे", "पवार",
                       "जोशी", "कुलकर्णी", "शाह", "भोसले"]

    candidates = []
    for idx, line in enumerate(lines):
        if any(h in line for h in honorifics) or any(s in line for s in common_surnames):
            candidates.append({
                "line_index": idx,
                "raw_text": line,
                "status": "unverified",
            })

    # De-duplicate
    seen = set()
    unique = []
    for c in candidates:
        if c["raw_text"] not in seen:
            seen.add(c["raw_text"])
            unique.append(c)
    return unique


# ---------- KEY FIELD EXTRACTION ----------

def extract_key_fields(text: str, doc_type: str) -> dict:
    key = {}

    if doc_type == "land_record":
        survey_pattern = r"(गट\s*क्र\.?\s*|गट\s*क्रमांक\s*|सर्वे\s*नं\.?\s*)([0-9]+)"
        m = re.search(survey_pattern, text)
        if m:
            key["survey_or_gat_no"] = {
                "value": m.group(2),
                "confidence": 0.8,
            }

        area_pattern = r"([0-9]+\s*(एकर|हे\.?|आर|गुंठे?))"
        m2 = re.search(area_pattern, text)
        if m2:
            key["area"] = {
                "value": m2.group(0),
                "confidence": 0.7,
            }

    key["possible_names"] = extract_possible_names(text)
    return key


# ---------- SIMPLE EXPLANATION ----------

def generate_simple_explanation(text: str, doc_type: str, key_fields: dict) -> dict:
    label_map = {
        "land_record": "Land / property record",
        "budget_or_accounts": "Budget / accounts register",
        "court_or_legal": "Court / legal document",
        "unknown": "Unknown type document",
    }
    doc_label = label_map.get(doc_type, "Unknown type document")

    owner_hint = None
    possible_names = key_fields.get("possible_names") or []
    if possible_names:
        owner_hint = possible_names[0]["raw_text"]

    survey_hint = key_fields.get("survey_or_gat_no", {}).get("value")
    area_hint = key_fields.get("area", {}).get("value")

    en = f"This appears to be a {doc_label}."
    mr = f"हा दस्त {doc_label} प्रकारचा दिसत आहे."

    if owner_hint:
        en += f" One important name mentioned is: '{owner_hint}'."
        mr += f" यात महत्त्वाचे नाव असे आहे: '{owner_hint}'."

    if survey_hint:
        en += f" It refers to survey/gat number {survey_hint}."
        mr += f" यात गट / सर्वे क्रमांक {survey_hint} असा उल्लेख आहे."

    if area_hint:
        en += f" The area mentioned is around {area_hint}."
        mr += f" जमिनीचे क्षेत्रफळ अंदाजे {area_hint} असे नमूद आहे."

    return {
        "summary_en": en,
        "summary_mr": mr,
        "note": "Names and legal details must always be manually checked with the original document.",
    }


# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="Saarthi Records (Prototype)", layout="wide")
st.title("📜 Saarthi Records – Old Document Reader (Prototype)")

st.write("Upload an old land / property / register document image. The app will try to:")
st.markdown("""
- Clean the image  
- Read Marathi/Hindi/English text  
- Guess document type (land, budget, court, etc.)  
- Extract key info (names, survey no., area)  
- Give a simple Marathi & English explanation  
""")

uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show original
    st.subheader("1️⃣ Original document")
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original image", use_column_width=True)

    # Preprocess
    st.subheader("2️⃣ Preprocessed (for OCR)")
    pre = preprocess_document(img)
    st.image(pre, caption="Preprocessed (binarized)", use_column_width=True, clamp=True)

    # OCR
    with st.spinner("Running OCR..."):
        ocr_text = run_ocr(pre, lang="mar+hin+eng")

    st.subheader("3️⃣ Raw OCR text")
    st.text_area("OCR output", ocr_text, height=200)

    # Doc type
    doc_info = detect_doc_type(ocr_text)
    st.subheader("4️⃣ Detected document type")
    st.write(doc_info)

    # Key fields
    key_fields = extract_key_fields(ocr_text, doc_info["type"])
    st.subheader("5️⃣ Key fields (draft)")
    st.json(key_fields)

    # Explanation
    explanation = generate_simple_explanation(ocr_text, doc_info["type"], key_fields)
    st.subheader("6️⃣ Simple explanation")
    st.markdown(f"**English:** {explanation['summary_en']}")
    st.markdown(f"**Marathi:** {explanation['summary_mr']}")
    st.caption(explanation["note"])

    st.warning(
        "⚠️ This is an AI prototype. Names and legal details may be wrong. "
        "Always verify with the original document and a legal expert."
    )
else:
    st.info("Upload an image to start.")
