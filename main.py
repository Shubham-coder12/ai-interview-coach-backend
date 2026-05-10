from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import tempfile
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTIONS = [
    "Tell me about yourself.",
    "What are your greatest strengths?",
    "What is your biggest weakness?",
    "Why do you want to work here?",
    "Tell me about a challenge you overcame.",
    "Where do you see yourself in 5 years?",
    "Why should we hire you?",
]

question_index = 0

@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...), question: str = Form(...)):
    global question_index
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        transcript = transcription.text

        analysis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an interview coach. Analyze the answer and return JSON with: score (0-100), feedback (constructive critique), strengths (what they did well), improvements (what to fix)"
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {transcript}\n\nScore on: Relevance (30pts), Structure (25pts), Specificity (25pts), Confidence (20pts)"
                }
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(analysis.choices[0].message.content)
        question_index += 1

        return {
            "transcript": transcript,
            "score": result["score"],
            "feedback": f"✅ Strengths: {result['strengths']}\n\n📈 Improvements: {result['improvements']}\n\n💡 Details: {result['feedback']}"
        }
    finally:
        os.unlink(tmp_path)

@app.post("/next-question")
async def next_question():
    global question_index
    
    if question_index < len(QUESTIONS):
        question = QUESTIONS[question_index]
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Generate a challenging behavioral interview question. Return just the question."}]
        )
        question = response.choices[0].message.content
    
    return {"question": question}

@app.get("/health")
async def health():
    return {"status": "alive"}
