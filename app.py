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

# --- LangChain Setup ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_completion_tokens=2000,
    timeout=30,
    streaming=True,
    max_retries=3
)

systemplate = PromptTemplate(
    template=f"""you are a hindu jain spiritual saint , follow all concepts of jainism , and some more knowledge is here {knowledge_base}
   
act as a spiritual guide and help people with your knowledge and wisdom , answer using nice stories and examples when required , 
answer in english and hindi as well if question is asked in that specific language . Given human (saint) like answers in paragraphs,
    """
)
final_sys_prompt = systemplate.format()

conv_memory = InMemoryChatMessageHistory()

def get_history():
    return conv_memory

langchain_chat = RunnableWithMessageHistory(llm, get_session_history=get_history)

INITIAL_BOT_GREETING = ""

def seed_system_prompt():
    global INITIAL_BOT_GREETING
    assembled = ""
    try:
        for piece in langchain_chat.stream(final_sys_prompt, config={"configurable": {"session_id": "default"}}):
            assembled += getattr(piece, "content", "")
    except Exception:
        assembled = "Welcome, seeker. How may I help you find peace today?"
    INITIAL_BOT_GREETING = assembled if assembled else "Welcome, seeker. How may I help you find peace today?"

# Start seeding in background
seed_thread = threading.Thread(target=seed_system_prompt, daemon=True)
seed_thread.start()

# --- Streaming class (KEPT EXACT ORIGINAL LOGIC) ---
@dataclass
class Piece:
    content: str

class ChatStreamer:
    def __init__(self):
        self._use_langchain = True
        self._langchain_chat = langchain_chat

    def stream(self, user_msg: str) -> Iterable[Piece]:
        if self._use_langchain:
            yield from self._langchain_stream(user_msg)
        else:
            yield from self._mock_stream(user_msg)

    def _langchain_stream(self, user_msg: str) -> Iterable[Piece]:
        try:
            stream = self._langchain_chat.stream(
                user_msg,
                config={"configurable": {"session_id": "default"}}
            )
            for chunk in stream:
                text = getattr(chunk, "content", None)
                if text:
                    yield Piece(text)
        except Exception as e:
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
    def generate():
        if not user_msg:
            yield "event: done\ndata:\n\n"
            return
        for piece in chat.stream(user_msg):
            if piece.content:
                yield f"data: {piece.content}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), headers=headers, mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)