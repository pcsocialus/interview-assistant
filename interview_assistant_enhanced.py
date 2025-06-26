
import streamlit as st
import openai
import sounddevice as sd
import numpy as np
import tempfile
import whisper
import scipy.io.wavfile as wav
import time

openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

st.set_page_config(page_title="AI Interview Assistant", layout="centered")
st.title("🎙️ AI Interview Assistant")

mode = st.radio("Choose Mode", ["🔹 Simple (1 Speaker)", "🔸 Simulated Diarization"])
tone = st.selectbox("Select Answer Tone", ["Professional", "Conversational", "Leadership", "Concise"])

SAMPLE_RATE = 44100
duration = st.slider("Recording duration (seconds)", min_value=2, max_value=15, value=5)

if st.button("🎤 Start Recording"):
    st.info("Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    st.success("Recording complete!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav.write(tmp_wav.name, SAMPLE_RATE, audio)
        tmp_audio_path = tmp_wav.name

    st.markdown("### Step 2: Transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe(tmp_audio_path)
    full_text = result["text"]
    st.success("Transcription Complete")

    st.markdown("### Transcribed Text:")
    st.write(full_text)

    if mode == "🔹 Simple (1 Speaker)":
        question_text = full_text
    else:
        sentences = full_text.split(". ")
        question_text = sentences[0] if sentences else full_text
        st.markdown("### Simulated Diarization:")
        st.write(f"🧑‍💼 **Interviewer**: {question_text}")
        if len(sentences) > 1:
            st.write(f"🧑‍💻 **Candidate (you)**: {' '.join(sentences[1:])}")

    prompt = f"The interviewer asked: '{question_text}'. Give a clear, strong, {tone.lower()} answer the interviewee can repeat confidently."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert interview coach generating confident and helpful answers."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response["choices"][0]["message"]["content"]

    st.markdown("### ✅ Suggested Answer:")
    st.markdown(f"<div style='background-color:#eefafc;padding:15px;border-radius:10px;font-size:18px'>{answer}</div>", unsafe_allow_html=True)

    # Teleprompter Mode
    st.markdown("### 📺 Teleprompter Mode:")
    scroll_speed = st.slider("Scroll speed (seconds per line)", 0.5, 3.0, 1.5)
    lines = answer.split(". ")
    for line in lines:
        st.write(f"> {line.strip()}")
        time.sleep(scroll_speed)

    # Export
    if st.button("📄 Export Q&A to .txt"):
        qa_text = f"""Interview Question:
{question_text}

AI-Generated Answer ({tone}):
{answer}
"""
        with open("interview_qa_output.txt", "w") as f:
            f.write(qa_text)
        with open("interview_qa_output.txt", "rb") as f:
            st.download_button(label="Download Q&A", data=f, file_name="interview_qa.txt")

else:
    st.info("Click the record button to begin.")
