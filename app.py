import streamlit as st
st.set_page_config(page_title="Text Understanding & Quiz Generator", layout="wide")

from PIL import Image, ImageFilter, ImageOps
import pytesseract
import language_tool_python
import random
import io
import re
import cv2
import numpy as np

try:
    from transformers import pipeline
    try:
        question_gen_pipeline = pipeline("e2e-qg")
    except Exception:
        question_gen_pipeline = None
except ImportError:
    question_gen_pipeline = None

try:
    tool = language_tool_python.LanguageToolPublicAPI('en-US')
except Exception:
    tool = None

def evaluate_text(text):
    if not tool:
        return 100, []
    matches = tool.check(text)
    issues = [match for match in matches]
    score = max(0, 100 - len(issues)*2)
    return score, issues

def preprocess_for_layered_ocr(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def extract_text_with_layout(pil_image):
    processed = preprocess_for_layered_ocr(pil_image)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    blocks = {}
    for i in range(n_boxes):
        text = data['text'][i].strip()
        block_num = data['block_num'][i]
        if text:
            blocks.setdefault(block_num, []).append(text)
    combined_blocks = [" ".join(blocks[b]) for b in sorted(blocks.keys())]
    return combined_blocks

def draw_text_boxes(pil_image, threshold=60):
    image = np.array(pil_image.convert("RGB"))
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > threshold:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return Image.fromarray(image)

def extract_keywords(text, num_keywords=20):
    stopwords = set([
        "the", "and", "is", "in", "to", "of", "a", "for", "on",
        "with", "as", "by", "an", "be", "at", "from", "that",
        "this", "it", "are", "was", "or", "but", "if", "or"
    ])
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if word not in stopwords and len(word) > 2:
            freq[word] = freq.get(word, 0) + 1
    sorted_keywords = sorted(freq, key=freq.get, reverse=True)
    return sorted_keywords[:num_keywords]

def generate_knowledge_quiz(text, min_questions=10):
    if question_gen_pipeline:
        try:
            generated = question_gen_pipeline(text)
            questions = []
            for item in generated[:min_questions]:
                question_text = item['question']
                answer_text = item['answer']
                keywords = extract_keywords(text)
                distractors = list(set([k for k in keywords if k.lower() != answer_text.lower()]))
                distractors = random.sample(distractors, k=3) if len(distractors) >= 3 else distractors + ["option1", "option2", "option3"]
                options = [answer_text] + distractors
                random.shuffle(options)
                questions.append({
                    "question": question_text,
                    "options": options,
                    "answer": answer_text
                })
            if questions:
                return questions
        except Exception:
            pass

    sentences = [s.strip() for s in re.split(r'[.?!]', text) if len(s.split()) > 5]
    keywords = extract_keywords(text)
    questions = []
    used_sentences = set()
    random.shuffle(sentences)
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence.lower() and sentence not in used_sentences:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                question_text = pattern.sub('______', sentence, count=1)
                correct_answer = keyword
                distractors = list(set([k for k in keywords if k != correct_answer]))
                distractors = random.sample(distractors, k=3) if len(distractors) > 3 else ["option1", "option2", "option3"]
                options = [correct_answer] + distractors
                random.shuffle(options)
                questions.append({
                    "question": f"Fill in the blank: {question_text}",
                    "options": options,
                    "answer": correct_answer
                })
                used_sentences.add(sentence)
                if len(questions) >= min_questions:
                    break
        if len(questions) >= min_questions:
            break

    while len(questions) < min_questions and keywords:
        keyword = keywords[len(questions) % len(keywords)]
        options = random.sample(keywords, k=4) if len(keywords) >= 4 else ["option1", "option2", "option3", "option4"]
        answer = options[0]
        random.shuffle(options)
        questions.append({
            "question": f"What is the meaning of '{keyword}'?",
            "options": options,
            "answer": answer
        })
    return questions[:min_questions]

def quiz_to_text(quiz_questions):
    output = io.StringIO()
    for i, q in enumerate(quiz_questions, 1):
        output.write(f"Q{i}: {q['question']}\n")
        for idx, opt in enumerate(q['options']):
            output.write(f"   {chr(65+idx)}. {opt}\n")
        output.write("\n")
    return output.getvalue()

def answers_to_text(quiz_questions):
    output = io.StringIO()
    for i, q in enumerate(quiz_questions, 1):
        correct_idx = q['options'].index(q['answer'])
        output.write(f"Q{i}: {chr(65+correct_idx)}. {q['answer']}\n")
    return output.getvalue()

st.title("🧠 Text Understanding & Quiz Generator")

tab1, tab2 = st.tabs(["📜 Text Input", "🖼️ Image Upload"])

with tab1:
    st.header("Text Input and Quiz Generation")
    user_text = st.text_area("Enter text for quiz generation:", height=200)

    if user_text.strip():
        st.subheader("🔍 Language Detection")
        try:
            from langdetect import detect
            lang = detect(user_text) if len(user_text.split()) >= 2 else "en"
        except Exception:
            lang = "en"
        st.success(f"Detected Language: {lang}")

        st.subheader("📊 Text Evaluation")
        score, issues = evaluate_text(user_text)
        st.metric(label="Grammar Score", value=f"{score}/100")
        if issues:
            st.write("Issues found (top 5):")
            for issue in issues[:5]:
                st.write(f"• {issue.message} (suggestion: {issue.replacements})")

        st.subheader("❓ Quiz Generator")
        st.info("Generating knowledge-level quiz from provided text... It may take a few seconds.")

        quiz_questions = generate_knowledge_quiz(user_text, min_questions=10)
        for i, q in enumerate(quiz_questions, 1):
            st.markdown(f"**Q{i}:** {q['question']}")
            for idx, opt in enumerate(q['options']):
                st.write(f"{chr(65+idx)}. {opt}")
            st.markdown("---")

        quiz_txt = quiz_to_text(quiz_questions)
        answers_txt = answers_to_text(quiz_questions)

        st.download_button(
            label="Download Quiz as Text",
            data=quiz_txt,
            file_name="quiz.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download Answers as Text",
            data=answers_txt,
            file_name="answers.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download Provided Text",
            data=user_text,
            file_name="provided_text.txt",
            mime="text/plain"
        )

# --- Main Tab 2 Block ---

with tab2:
    st.header("🖼️ Image Upload and Text Extraction")
    uploaded_image = st.file_uploader("Upload an image (png, jpg, jpeg):", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Extracting text with layout-aware layered OCR..."):
            blocks = extract_text_with_layout(image)

        st.subheader("📝 Extracted Text Blocks")
        for i, block in enumerate(blocks, 1):
            st.text_area(f"Block {i}", block, height=100)

        extracted_text = "\n".join(blocks)

        st.subheader("📊 Extracted Text Evaluation")
        score, issues = evaluate_text(extracted_text)
        st.metric(label="Grammar Score", value=f"{score}/100")
        if issues:
            st.write("Issues found (top 5):")
            for issue in issues[:5]:
                st.write(f"• {issue.message} (suggestion: {issue.replacements})")

        st.subheader("❓ Quiz Generator from Extracted Text")
        st.info("Generating knowledge-level quiz from extracted content... It may take a few seconds.")

        quiz_questions = generate_knowledge_quiz(extracted_text, min_questions=10)
        for i, q in enumerate(quiz_questions, 1):
            st.markdown(f"**Q{i}:** {q['question']}")
            for idx, opt in enumerate(q['options']):
                st.write(f"{chr(65+idx)}. {opt}")
            st.markdown("---")

        quiz_txt = quiz_to_text(quiz_questions)
        answers_txt = answers_to_text(quiz_questions)

        st.download_button(
            label="📥 Download Quiz as Text",
            data=quiz_txt,
            file_name="quiz.txt",
            mime="text/plain"
        )
        st.download_button(
            label="📥 Download Answers as Text",
            data=answers_txt,
            file_name="answers.txt",
            mime="text/plain"
        )
        st.download_button(
            label="📥 Download Extracted Text",
            data=extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

        # Optional: OCR bounding box view
        if st.checkbox("📦 Show OCR Bounding Boxes"):
            boxed_image = draw_text_boxes(image)
            st.image(boxed_image, caption="OCR Text Regions", use_column_width=True)
