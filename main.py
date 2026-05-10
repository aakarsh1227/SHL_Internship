import json
import httpx
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import your custom data and models
from catalog import CATALOG
from models import ChatRequest, ChatResponse

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED: Ensure variable name matches usage in the function
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

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

    # 2. Extract keywords to filter the large catalog
    user_query = request.messages[-1].content.lower()
    
    # Keyword filtering prevents prompt overflow and 503 errors
    relevant_catalog = [
        item for item in CATALOG 
        if any(word in item["name"].lower() or word in item.get("description", "").lower() 
               for word in user_query.split())
    ]
    
    # Fallback to avoid empty context
    if not relevant_catalog:
        relevant_catalog = CATALOG[:10]

    # 3. Build Lean System Prompt
    system_prompt = (
        "You are an SHL Product Expert. Based on the user query, suggest 1-3 products "
        "ONLY from this relevant list. Return a friendly reply and a valid JSON list "
        "of recommendations.\n\nRelevant Catalog:\n" + str(relevant_catalog)[:4000]
    )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json={"inputs": str(formatted_messages), "parameters": {"max_new_tokens": 500}},
                timeout=30.0
            )

        if response.status_code == 503:
            return ChatResponse(
                reply="AI service is warming up. Try again in 5s.", 
                recommendations=[], 
                end_of_conversation=False
            )

        # 4. Robust Response Parsing
        ai_output = response.json()[0]['generated_text']
        
        # Extract JSON list using regex
        match = re.search(r'\[\s*{.*}\s*\]', ai_output, re.DOTALL)
        recommendations = []
        if match:
            try:
                recommendations = json.loads(match.group())
            except:
                recommendations = []
            
        reply = ai_output.split("[")[0].strip() if "[" in ai_output else ai_output
        # Clean up any leftover prompt tags from Mistral
        reply = reply.replace("<s>", "").replace("[INST]", "").strip()

        return ChatResponse(
            reply=reply,
            recommendations=recommendations[:3],
            end_of_conversation=len(recommendations) > 0
        )

    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}", recommendations=[], end_of_conversation=False)