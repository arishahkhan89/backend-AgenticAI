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
import httpx
from gtts import gTTS
import os, json
from pathlib import Path
from agents import (
    Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel,
    RunContextWrapper, function_tool, trace
)
from agents.run import RunConfig

# ---------------- ENV & CONFIG ----------------
load_dotenv()

# ---------------- FASTAPI APP ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    print("🔹 greet_user tool called")  # Debug
    return "Hello! Welcome to Smart Agri. Have a great day. Farm-Genie here.. What would you like to ask?"

@function_tool
async def show_irrigation_advice(wrapper: RunContextWrapper[None]) -> dict:
    print("🔹 show_irrigation_advice tool called")  # Debug
    return {"message": "Ok, I am providing you the irrigation agent you can ask him your query.", "redirect": True, "redirect_url": "/irrigation_advice"}

@function_tool
async def get_weather_and_soil(location: str) -> dict:
    try:
        print(f"🔹 Fetching weather for location: {location}")
        weather_url = f"https://api.weatherapi.com/v1/current.json?key={os.getenv('WEATHER_API_KEY')}&q={location}"
        async with httpx.AsyncClient() as http:
            weather_res = await http.get(weather_url)
        print(f"🔹 Weather API status code: {weather_res.status_code}")
        weather = weather_res.json()
        print(f"🔹 Weather response: {weather}")

        dummy_path = Path(__file__).parent / "dummy_data.json"
        with open(dummy_path, "r") as f:
            soil_records = json.load(f)
        print(f"🔹 Loaded dummy soil records: {soil_records}")

        soil_data = next(
            (record["soil"] for record in soil_records if record["location"].lower() == location.lower()), 
            None
        )
        print(f"🔹 Soil data found: {soil_data}")

        if not soil_data:
            soil_data = {
                "moisture": 25,
                "ph": 7.0,
                "nitrogen": "medium",
                "phosphorus": "medium",
                "potassium": "medium"
            }
            print("🔹 Using fallback soil data")

        moisture = soil_data["moisture"]
        temp_c = weather["current"]["temp_c"]

        if moisture < 30:
            advice = "You should irrigate today with more water. Best time is evening." if temp_c >= 30 else "You should irrigate today with moderate water. Best time is evening."
        else:
            advice = "No irrigation needed today."

        return {
            "weather": weather,
            "soil": soil_data,
            "advice": advice
        }

    except Exception as e:
        print(f"❌ Error in get_weather_and_soil: {e}")
        return {"error": str(e)}

# ============================ Agents ========================================

greeting_agent = Agent[None](
    name="Greeting Assistant",
    instructions="""You are a Greeting Assistant. Handle greetings only.""",
    model=model,
    tools=[greet_user]
)

irrigation_agent = Agent[None](
    name="Irrigation Assistant",
    instructions="""Handle generic irrigation requests and redirect to advice.""",
    model=model,
    tools=[show_irrigation_advice]
)

irrigation_advice_agent = Agent[None](
    name="Smart Irrigation Advice Agent",
    instructions="""Handle detailed irrigation advice with weather and soil data.""",
    model=model,
    tools=[get_weather_and_soil]
)

main_agent = Agent[None](
    name="Main Agriculture Agent",
    instructions="""Route greetings and irrigation queries to appropriate agents.""",
    model=model,
    tools=[],
    handoffs=[greeting_agent, irrigation_agent, irrigation_advice_agent]
)

# ---------------- AGENT ROUTE ----------------
@app.post("/agent")
async def ask_agent(question: str = Form(None), audio: UploadFile = File(None)):
    try:
        if audio:
            print("🔹 Audio received, converting to text")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(await audio.read())
                temp_audio_path = temp_audio.name

            temp_wav_path = tempfile.mktemp(suffix=".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_audio_path, temp_wav_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav_path) as source:
                audio_data = recognizer.record(source)
                question = recognizer.recognize_google(audio_data)
                print(f"🔹 Recognized question from audio: {question}")

            os.remove(temp_audio_path)
            os.remove(temp_wav_path)

        if not question:
            raise HTTPException(status_code=400, detail="No question or audio provided.")

        print(f"🔹 User question: {question}")
        with trace("Agentic Agriculture"):  
            result = await Runner.run(main_agent, question, run_config=run_config, context=None)
            final_output = result.final_output
            print(f"🔹 Agent final output: {final_output}")

            if isinstance(final_output, dict):
                response_data = {
                    "message": final_output.get("message", ""),
                    "redirect": final_output.get("redirect", False),
                    "redirect_url": final_output.get("redirect_url", None)
                }
            elif isinstance(final_output, str):
                response_data = {
                    "message": final_output,
                    "redirect": False,
                    "redirect_url": None
                }
            else:
                response_data = {"message": "Unexpected response", "redirect": False, "redirect_url": None}

            question_lower = question.lower()
            if "irrigation" in question_lower or "tell me irrigation" in question_lower or "mujhe irrigation" in question_lower:
                response_data["redirect"] = True
                response_data["redirect_url"] = "/irrigation_advice"

        tts = gTTS(response_data["message"], lang="en")
        audio_file_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".mp3")
        tts.save(audio_file_path)
        response_data["audio_url"] = f"/agent-audio?file={os.path.basename(audio_file_path)}"

        return response_data

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-audio")
async def get_agent_audio(file: str):
    file_path = os.path.join(tempfile.gettempdir(), file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=file)

# ---------------- SERVER ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
