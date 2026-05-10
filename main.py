import json
import httpx
import os
import re
from catalog import CATALOG
from pydantic import BaseModel
from typing import List, Optional
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
    # 1. Enforce the 8-turn limit
    if len(request.messages) > 8:
        return ChatResponse(
            reply="Conversation limit reached. How else can I help?",
            recommendations=[],
            end_of_conversation=True
        )

    # 2. Extract keywords from the user's latest message to filter the catalog
    user_query = request.messages[-1].content.lower()
    
    # Filter catalog: Only include items where the name or description matches a keyword
    # This reduces the prompt size from 100 items to ~5-10 items
    relevant_catalog = [
        item for item in CATALOG 
        if any(word in item["name"].lower() or word in item["description"].lower() 
               for word in user_query.split())
    ]
    
    # Fallback: If no keywords match, send a small default sample so the AI isn't blind
    if not relevant_catalog:
        relevant_catalog = CATALOG[:10]

    # 3. Build a Lean System Prompt
    system_prompt = (
        "You are an SHL Product Expert. Based on the user query, suggest 1-3 products "
        "ONLY from this relevant list. Return a friendly reply and a valid JSON list "
        "of recommendations.\n\nRelevant Catalog:\n" + str(relevant_catalog)[:4000] # Cap length
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json={"inputs": str(messages), "parameters": {"max_new_tokens": 500}},
                timeout=20.0
            )

        if response.status_code == 503:
            return ChatResponse(reply="AI service is warming up. Try again in 5s.", recommendations=[], end_of_conversation=False)

        # Parse AI response
        raw_text = response.json()[0]['generated_text']
        
        # Simple extraction of reply and recommendations (ensure your parsing logic is robust here)
        # For the sake of this fix, we assume the AI returns valid JSON as requested
        import json
        import re
        
        # Look for JSON array in the output
        match = re.search(r'\[\s*{.*}\s*\]', raw_text, re.DOTALL)
        recommendations = []
        if match:
            recommendations = json.loads(match.group())
            
        reply = raw_text.split("[")[0].strip() if "[" in raw_text else raw_text

        return ChatResponse(
            reply=reply,
            recommendations=recommendations[:3],
            end_of_conversation=len(recommendations) > 0
        )

    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}", recommendations=[], end_of_conversation=False)