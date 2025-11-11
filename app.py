# main app.py 
import os, time, threading
from dataclasses import dataclass
from typing import Iterable
from flask import Flask, request, render_template, Response, stream_with_context, jsonify

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from text import knowledge_base

from dotenv import load_dotenv

load_dotenv()

# Access environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# --- LangChain Setup ---
llm = ChatOpenAI(
    model="gpt-4o-mini",  # OpenRouter model name
    temperature=0.3,
    max_completion_tokens=2000,
    timeout=30,
    streaming=True,
    max_retries=3,
    openai_api_key=openrouter_api_key,  # Use OpenRouter key
    openai_api_base=openai_base_url
)

systemplate = PromptTemplate(
    template=f"""You are a wise and compassionate Jain spiritual saint (a Hindu-Jain monk) who follows and embodies the core principles of Jainism — Ahimsa (non-violence), Satya (truth), Asteya (non-stealing), Brahmacharya (self-control), and Aparigraha (non-possessiveness).

You act as a spiritual guide, offering peaceful, thoughtful, and enlightening responses based on Jain philosophy and related Hindu wisdom. You explain concepts with gentle reasoning, real-life analogies, and inspiring short stories whenever needed.

Your tone should be calm, kind, and deeply reflective — like a human saint speaking from experience and compassion. You answer primarily in **English**, but if the question is asked in **Hindi**, respond naturally in **Hindi** with the same grace and depth.

Structure each response as follows:
1. **Teach clearly and calmly**, explaining the meaning or principle first.  
2. **Offer a deeper explanation**, linking it to Jain philosophy, karma, soul (atma), and right conduct (dharma).  
3. **Conclude with an illustrative story or example**, to make the wisdom relatable and easy to remember.  

Use this additional knowledge when needed:  
{knowledge_base}

Speak as a guide who helps people find peace, awareness, and self-purification through reflection and right understanding.
"""
)
final_sys_prompt = systemplate.format()

# Session storage - should be per-user in production
# For now using simple dict, consider Redis for multi-instance deployments
session_store = {}

def get_history(session_id: str = "default"):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

langchain_chat = RunnableWithMessageHistory(llm, get_session_history=get_history)

INITIAL_BOT_GREETING = ""

def seed_system_prompt():
    global INITIAL_BOT_GREETING
    assembled = ""
    try:
        for piece in langchain_chat.stream(final_sys_prompt, config={"configurable": {"session_id": "system"}}):
            assembled += getattr(piece, "content", "")
    except Exception as e:
        print(f"Seeding error: {e}")
        assembled = "Welcome, seeker. How may I help you find peace today?"
    INITIAL_BOT_GREETING = assembled if assembled else "Welcome, seeker. How may I help you find peace today?"

# Start seeding in background
seed_thread = threading.Thread(target=seed_system_prompt, daemon=True)
seed_thread.start()

# --- Streaming class ---
@dataclass
class Piece:
    content: str

class ChatStreamer:
    def __init__(self):
        self._use_langchain = True
        self._langchain_chat = langchain_chat

    def stream(self, user_msg: str, session_id: str = "default") -> Iterable[Piece]:
        if self._use_langchain:
            yield from self._langchain_stream(user_msg, session_id)
        else:
            yield from self._mock_stream(user_msg)

    def _langchain_stream(self, user_msg: str, session_id: str) -> Iterable[Piece]:
        try:
            stream = self._langchain_chat.stream(
                user_msg,
                config={"configurable": {"session_id": session_id}}
            )
            for chunk in stream:
                text = getattr(chunk, "content", None)
                if text:
                    yield Piece(text)
        except Exception as e:
            print(f"Stream error: {e}")
            yield Piece(f"[Error: {e}]")

    def _mock_stream(self, user_msg: str) -> Iterable[Piece]:
        reply = f"You said: {user_msg}\nThis is a mock streamed reply."
        for c in reply:
            yield Piece(c)
            time.sleep(0.01)

# --- Flask app ---
app = Flask(__name__, template_folder="templates")
chat = ChatStreamer()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    # Wait for seeding to complete (with timeout)
    seed_thread.join(timeout=12)
    return render_template("chat.html", initial_greeting=INITIAL_BOT_GREETING)

@app.get("/api/stream")
def chat_stream():
    user_msg = (request.args.get("message") or "").strip()
    session_id = request.args.get("session_id", "default")
    
    def generate():
        if not user_msg:
            yield "event: done\ndata:\n\n"
            return
        for piece in chat.stream(user_msg, session_id):
            if piece.content:
                yield f"data: {piece.content}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
    
    headers = {
        "Cache-Control": "no-cache", 
        "Connection": "keep-alive", 
        "X-Accel-Buffering": "no"
    }
    return Response(stream_with_context(generate()), headers=headers, mimetype="text/event-stream")

# Health check endpoint for Render
@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":

    app.run(debug=True)