import json
import httpx
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- SCHEMAS ---
class Message(BaseModel):
    role: str
    content: str

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# --- APP SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
# Use the generic Inference API URL which automatically routes to the best available instance
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# --- DATA ---
# (Using a sample here - keep your full catalog.py if it's working, 
# but if catalog.py had errors, paste the JSON list here instead)
try:
    from catalog import CATALOG
except ImportError:
    CATALOG = [
        {"name": "OPQ32r", "link": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "description": "Personality test for managers", "keys": ["Personality"]}
    ]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        return ChatResponse(reply="How can I help?", recommendations=[], end_of_conversation=False)

    user_query = request.messages[-1].content.lower()
    
    # Filter catalog
    relevant = [
        item for item in CATALOG 
        if any(word in item.get("name", "").lower() or word in item.get("description", "").lower() 
               for word in user_query.split())
    ]
    
    context_data = relevant[:5] if relevant else CATALOG[:5]

    system_prompt = (
        "You are an SHL Assistant. Suggest 1-3 relevant products from the list below. "
        "Return a JSON array of objects with keys 'name', 'url', and 'test_type'.\n\n"
        f"Catalog: {json.dumps(context_data)}"
    )

    formatted_msgs = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        formatted_msgs.append({"role": msg.role, "content": msg.content})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json={"inputs": str(formatted_msgs), "parameters": {"max_new_tokens": 500}},
                timeout=30.0
            )

        if response.status_code == 503:
            return ChatResponse(reply="AI is warming up. Retry in 5s.", recommendations=[], end_of_conversation=False)

        res_data = response.json()
        # Handle different HF response formats
        ai_text = res_data[0].get('generated_text', '') if isinstance(res_data, list) else str(res_data)

        # Regex to find the JSON block
        match = re.search(r'\[\s*{.*}\s*\]', ai_text, re.DOTALL)
        recs = []
        if match:
            try:
                # Clean the match for any trailing chars
                clean_json = match.group().replace("'", '"')
                recs = json.loads(clean_json)
            except:
                recs = []

        clean_reply = ai_text.split("[")[0].replace("<s>", "").replace("[INST]", "").strip()

        return ChatResponse(
            reply=clean_reply if clean_reply else "Here are my recommendations:",
            recommendations=[
                Recommendation(name=r.get("name", "Test"), url=r.get("url", "#"), test_type=r.get("test_type", "General"))
                for r in recs
            ][:3],
            end_of_conversation=len(recs) > 0
        )

    except Exception as e:
        return ChatResponse(reply=f"Service busy. Please try again.", recommendations=[], end_of_conversation=False)