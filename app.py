import streamlit as st
st.set_page_config(page_title="Text Understanding & Quiz Generator", layout="wide")

from PIL import Image, ImageFilter, ImageOps
import pytesseract
import language_tool_python
import random
import io
import re
# Optional: HuggingFace transformers pipeline for question generation
try:
    from transformers import pipeline
    try:
        question_gen_pipeline = pipeline("e2e-qg")
    except Exception:
        question_gen_pipeline = None
except ImportError:
    question_gen_pipeline = None

# --- Grammar Evaluation ---
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

# --- Image Preprocessing ---
def preprocess_image(image):
    """
    Preprocess the input PIL image for better OCR accuracy.
    Steps:
    - Convert to grayscale
    - Increase contrast
    - Apply binary thresholding
    - Remove noise with median filter
    - Resize small images for better OCR
    """
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    threshold = 140
    bw = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
    bw = bw.filter(ImageFilter.MedianFilter(size=3))
    if bw.width < 300:
        new_size = (bw.width * 2, bw.height * 2)
        bw = bw.resize(new_size, Image.Resampling.LANCZOS)
    return bw

# --- Text Extraction ---
def extract_text_from_image(image):
    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed)
    return text

def extract_text_with_layout(image):
    preprocessed = preprocess_image(image)
    data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    blocks = {}
    for i in range(n_boxes):
        block_num = data['block_num'][i]
        text = data['text'][i].strip()
        if text:
            blocks.setdefault(block_num, []).append(text)
    return [" ".join(blocks[b]) for b in sorted(blocks.keys())]

# --- Keyword Extraction ---
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

# --- Quiz Generation ---
def generate_knowledge_quiz(text, min_questions=10):
    """
    Generate quiz questions using transformers pipeline if available,
    fallback to keyword & sentence based fill-in-the-blank questions.
    """
    if question_gen_pipeline:
        try:
            generated = question_gen_pipeline(text)
            questions = []
            for item in generated[:min_questions]:
                question_text = item['question']
                answer_text = item['answer']
                keywords = extract_keywords(text)
                distractors = [k for k in keywords if k.lower() != answer_text.lower()]
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
            pass  # fall back if model fails

    # Fallback fill-in-the-blank keyword based question generation
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
                distractors = random.sample(
                    [k for k in keywords if k != correct_answer], k=3) if len(keywords) > 3 else ["option1", "option2", "option3"]
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
    # Fill remaining with simple keyword definition style questions if needed
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

# --- Text representation of quiz and answers for download ---
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

# --- Streamlit UI ---
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

with tab2:
    st.header("Image Upload and Text Extraction")
    uploaded_image = st.file_uploader("Upload an image (png, jpg, jpeg):", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text with layout-aware analysis..."):
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
            label="Download Extracted Text",
            data=extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )