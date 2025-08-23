import os
import tempfile
from dataclasses import dataclass
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import traceback
import subprocess
import speech_recognition as sr
from gtts import gTTS
from agents import (
    Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel,
    RunContextWrapper, function_tool,trace
)
from agents.run import RunConfig

# ---------------- ENV & CONFIG ----------------
load_dotenv()


# ---------------- FASTAPI APP ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AGENTS SETUP ----------------
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)
run_config = RunConfig(model=model, model_provider=external_client)


# =====================================Function Tools ========================

@function_tool
async def greet_user(wrapper: RunContextWrapper[None]) -> str:
    return "Hello!  Welcome to Smart Agri. Have a great day . Farm-Genie here.. What would you like to ask?"

# ============================Agents========================================

greeting_agent = Agent[None](
    name="Greeting Assistant",
    instructions="""
Always call the tool `greet_user` when you receive any greeting like "hello", "hi", "salam".
Do not reply by yourself. You must call the tool.
""",
    model=model,
    tools=[greet_user]
)

main_agent = Agent[None](
    name="Main Restaurant Agent",
    instructions="""
You are the main restaurant assistant.
- If the user says "hello", "hi", "salam", you must handoff to Greeting Assistant. 
- Greeting Assistant must then call `greet_user`.

""",
    model=model,
    tools=[],
    handoffs=[greeting_agent]
)


# # ---------------- AGENT ROUTE ----------------
# @app.post("/agent")
# async def ask_agent(question: str = Form(None), audio: UploadFile = File(None)):
#     try:
#         # (🎤 audio to text conversion)
#         if audio:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
#                 temp_audio.write(await audio.read())
#                 temp_audio_path = temp_audio.name

#             temp_wav_path = tempfile.mktemp(suffix=".wav")
#             subprocess.run(
#                 ["ffmpeg", "-y", "-i", temp_audio_path, temp_wav_path],
#                 stdout=subprocess.PIPE, stderr=subprocess.PIPE
#             )

#             recognizer = sr.Recognizer()
#             with sr.AudioFile(temp_wav_path) as source:
#                 audio_data = recognizer.record(source)
#                 question = recognizer.recognize_google(audio_data)

#             os.remove(temp_audio_path)
#             os.remove(temp_wav_path)

#         if not question:
#             raise HTTPException(status_code=400, detail="No question or audio provided.")

#         # 🟢 Agent ko run karo
#         with trace("Agentic Agriculture"):  
#             result = await Runner.run(main_agent, question, run_config=run_config, context=None)
#             final_output = result.final_output

#             # Agar dict aayi (tool ka structured response)
#             if isinstance(final_output, dict):
#                 response_data = {
#                     "message": final_output.get("message", ""),
#                     "redirect": final_output.get("redirect", False),
#                     "redirect_url": final_output.get("redirect_url", None)
#                 }

#             # Agar sirf string aayi
#             elif isinstance(final_output, str):
#                 response_data = {
#                     "message": final_output,
#                     "redirect": False,
#                     "redirect_url": None
#                 }

#             # 👇 Yahan force redirect logic lagao
#             question_lower = question.lower()

#             # Menu ke liye
#             if "menu" in question_lower or "show menu" in question_lower or "mujhe menu" in question_lower:
#                 response_data["redirect"] = True
#                 response_data["redirect_url"] = "/menu"

#             # Order confirmation ke liye
#             elif (
#                 "order" in question_lower 
#                 or "order confirmation" in question_lower 
#                 or "my order" in question_lower 
#                 or "mera order" in question_lower
#             ):
#                 response_data["redirect"] = True
#                 response_data["redirect_url"] = "/order_summary"

#             # Agar unknown type ho
#             else:
#                 response_data = {
#                     "message": str(final_output),
#                     "redirect": False,
#                     "redirect_url": None
#                 }


#         # 🎤 TTS - hamesha message ko speech banao
#         tts = gTTS(response_data["message"], lang="en")
#         audio_file_path = os.path.join(
#             tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".mp3"
#         )
#         tts.save(audio_file_path)

#         response_data["audio_url"] = f"/agent-audio?file={os.path.basename(audio_file_path)}"

#         return response_data

#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/agent-audio")
# async def get_agent_audio(file: str):
#     file_path = os.path.join(tempfile.gettempdir(), file)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="Audio file not found")
#     return FileResponse(file_path, media_type="audio/mpeg", filename=file)


# ---------------- SERVER ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
