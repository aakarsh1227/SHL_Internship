import json
import os
import re
from catalog import CATALOG
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient
from dotenv import load_dotenv
from models import ChatRequest, ChatResponse

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# Remove the Markdown brackets and links
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# 1. Define Catalog OUTSIDE the function
CATALOG = [
    {"name": "Numerical Reasoning", "url": "https://www.shl.com/shl-numerical-reasoning-test/", "test_type": "K", "description": "Measures the ability to work with numbers and data."},
    {"name": "Verbal Reasoning", "url": "https://www.shl.com/verbal-reasoning-test/", "test_type": "K", "description": "Measures the ability to evaluate written information."},
    {"name": "Inductive Reasoning", "url": "https://www.shl.com/inductive-reasoning-test/", "test_type": "K", "description": "Measures the ability to find patterns and solve problems."},
    {"name": "Deductive Reasoning", "url": "https://www.shl.com/deductive-reasoning-test/", "test_type": "K", "description": "Measures the ability to draw logical conclusions."},
    {"name": "Personality Questionnaire (OPQ)", "url": "https://www.shl.com/occupational-personality-questionnaire/", "test_type": "P", "description": "The OPQ32 measures behavioral style at work."},
    {"name": "Situational Judgement", "url": "https://www.shl.com/situational-judgement-test/", "test_type": "S", "description": "Evaluates how you handle workplace scenarios."},
    {"name": "Calculation Test", "url": "https://www.shl.com/calculation-test/", "test_type": "K", "description": "Measures basic mathematical calculation speed and accuracy."},
    {"name": "Checking Test", "url": "https://www.shl.com/checking-test/", "test_type": "K", "description": "Measures the ability to quickly and accurately compare data."},
    {"name": "Motivation Questionnaire", "url": "https://www.shl.com/motivation-questionnaire/", "test_type": "P", "description": "Identifies what drives and motivates an individual at work."},
    {"name": "Java 8 (New)", "url": "https://www.shl.com/java-test/", "test_type": "K", "description": "Specific technical assessment for Java programming skills."}
]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        turn_count = len(request.messages)
        is_last_turn = turn_count >= 7 

        # 2. Clean System Instruction
        system_instruction = f"""You are an SHL Product Expert. 
CATALOG: {json.dumps(CATALOG)}

RULES:
1. If query is vague, ask for role/seniority.
2. Recommend only from the catalog provided.
3. Use exact names and URLs.
4. Output MUST be ONLY valid JSON. No conversational filler.
"""
        if is_last_turn:
            system_instruction += "\nCRITICAL: Final turn. Provide recommendations now."

        # 3. Build History
        # Build History using Mistral-specific tags
        user_history = ""
        for msg in request.messages:
            if msg.role == "user":
                user_history += f" [INST] {msg.content} [/INST] "
            else:
                user_history += f" {msg.content} "
            
        full_prompt = f"<s>[INST] {system_instruction} [/INST] {user_history} assistant:"

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        async with AsyncClient(timeout=60.0) as client: 
            response = await client.post(
                HF_API_URL,
                headers=headers,
                json={
                    "inputs": full_prompt,
                    "parameters": {"max_new_tokens": 500, "temperature": 0.1, "return_full_text": False},
                    "options": {"wait_for_model": True}
                }
            )

            if response.status_code != 200:
                return {"reply": "AI service is warming up. Try again in 5s.", "recommendations": [], "end_of_conversation": False}

            result = response.json()
            gen_text = result[0]['generated_text'] if isinstance(result, list) else str(result)

            # 4. Safer Regex extraction
            match = re.search(r'\{.*\}', gen_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                has_recs = len(data.get("recommendations", [])) > 0
                return {
                    "reply": data.get("reply", "Here are your recommendations."),
                    "recommendations": data.get("recommendations", []),
                    "end_of_conversation": True if (is_last_turn or has_recs) else data.get("end_of_conversation", False)
                }
            
            return {"reply": "I couldn't generate a proper response. Please refine your request.", "recommendations": [], "end_of_conversation": is_last_turn}

    except Exception as e:
        # Check your terminal for this print output to see the REAL error
        print(f"DEBUG ERROR: {str(e)}")
        return {"reply": "Internal processing error.", "recommendations": [], "end_of_conversation": False}