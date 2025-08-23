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
    return "Hello! Welcome to Smart Agri. Have a great day. Farm-Genie here.. What would you like to ask?"

@function_tool
async def show_irrigation_advice(wrapper: RunContextWrapper[None]) -> dict:
    return {"message": "Ok, I am providing you the irrigation agent you can ask him your query.", "redirect": True, "redirect_url": "/irrigation_advice"}

@function_tool
async def get_weather_and_soil(location: str) -> dict:
    """
    Fetch weather from WeatherAPI.com and soil data from dummy_data.json.
    Generate irrigation advice based on soil moisture and weather.
    """
    try:
        # 1️⃣ Weather fetch
        weather_url = f"https://api.weatherapi.com/v1/current.json?key={os.getenv('WEATHER_API_KEY')}&q={location}"
        async with httpx.AsyncClient() as http:
            weather_res = await http.get(weather_url)
        weather = weather_res.json()

        # 2️⃣ Load soil data from dummy_data.json
        dummy_path = Path(__file__).parent / "dummy_data.json"   # ✅ fixed
        with open(dummy_path, "r") as f:
            soil_records = json.load(f)

        # 3️⃣ Find soil data for this location
        soil_data = next(
            (record["soil"] for record in soil_records if record["location"].lower() == location.lower()), 
            None
        )

        # Fallback agar location na mile
        if not soil_data:
            soil_data = {
                "moisture": 25,
                "ph": 7.0,
                "nitrogen": "medium",
                "phosphorus": "medium",
                "potassium": "medium"
            }

        # 4️⃣ Generate irrigation advice
        moisture = soil_data["moisture"]
        temp_c = weather["current"]["temp_c"]

        if moisture < 30:
            if temp_c >= 30:
                advice = "You should irrigate today with more water. Best time is evening."
            else:
                advice = "You should irrigate today with moderate water. Best time is evening."
        else:
            advice = "No irrigation needed today."

        # 5️⃣ Return combined data
        return {
            "weather": weather,
            "soil": soil_data,
            "advice": advice
        }

    except Exception as e:
        return {"error": str(e)}

# ============================ Agents ========================================

greeting_agent = Agent[None](
    name="Greeting Assistant",
    instructions="""
You are a Greeting Assistant.
Your ONLY job is to handle greetings such as:
- "hello", "hi", "hey", "salam", "assalamualaikum", "good morning", "good evening".

👉 When you detect a greeting, ALWAYS call the greet_user tool.  
👉 Do NOT generate your own response.  
👉 Never answer irrigation, weather, or soil-related queries. Those must be handed off to other agents.
""",
    model=model,
    tools=[greet_user]
)

irrigation_agent = Agent[None](
    name="Irrigation Assistant",
    instructions="""If the user asks about "irrigation", "tell me irrigation", "mujhe irrigation" then you must call the tool show_irrigation_advice and redirect to irrigation_advice url.
    Just follow the instructions and do not answer by yourself""",
    model=model,
    tools=[show_irrigation_advice]
)

irrigation_advice_agent = Agent[None](
    name="Smart Irrigation Advice Agent",
    instructions="""
You are a Smart Irrigation Assistant 🤖.

🎯 Your job:
1. Whenever a user asks about irrigation, watering crops, 
   weather for farming, or soil conditions, ALWAYS call the get_weather_and_soil tool.
2. Extract the location from the user query. The location is usually mentioned as a city name (e.g., "Lahore", "Karachi").
3. Use both WEATHER + SOIL data returned by the tool to decide:
   a. Should the farmer irrigate today? (Yes/No)
   b. How much water is recommended (light / moderate / heavy irrigation).
   c. Best time of day for irrigation (morning / evening).
4. Provide the advice as a clear, concise message to the user.
5. Optionally, summarize the current temperature and soil moisture for context.
6. ALWAYS call get_weather_and_soil(location) first to get accurate data before giving advice.
""",
    model=model,
    tools=[get_weather_and_soil]
)

main_agent = Agent[None](
    name="Main Agriculture Agent",
    instructions="""
1. If the user greets (hello, hi, salam, etc.) → HANDOFF to the Greeting Assistant.
2. If the user asks about "irrigation", "tell me irrigation", "mujhe irrigation", you must handoff to Irrigation Assistant. 
3. Irrigation Assistant must then call the tool `show_irrigation_advice`.
4. If the user asks about irrigation with location provided → HANDOFF to the Smart Irrigation Advice Agent.
""",
    model=model,
    tools=[],
    handoffs=[greeting_agent, irrigation_agent, irrigation_advice_agent]
)

# ---------------- AGENT ROUTE ----------------
@app.post("/agent")
async def ask_agent(question: str = Form(None), audio: UploadFile = File(None)):
    try:
        # (🎤 audio to text conversion)
        if audio:
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

            os.remove(temp_audio_path)
            os.remove(temp_wav_path)

        if not question:
            raise HTTPException(status_code=400, detail="No question or audio provided.")

        # Run agent
        with trace("Agentic Agriculture"):  
            result = await Runner.run(main_agent, question, run_config=run_config, context=None)
            final_output = result.final_output

            # Agar dict aayi (tool ka structured response)
            if isinstance(final_output, dict):
                response_data = {
                    "message": final_output.get("message", ""),
                    "redirect": final_output.get("redirect", False),
                    "redirect_url": final_output.get("redirect_url", None)
                }
            # Agar sirf string aayi
            elif isinstance(final_output, str):
                response_data = {
                    "message": final_output,
                    "redirect": False,
                    "redirect_url": None
                }
            else:
                response_data = {"message": "Unexpected response", "redirect": False, "redirect_url": None}

            # Manual redirect check
            question_lower = question.lower()
            if "irrigation" in question_lower or "tell me irrigation" in question_lower or "mujhe irrigation" in question_lower:
                response_data["redirect"] = True
                response_data["redirect_url"] = "/irrigation_advice"

        # 🎤 TTS - hamesha message ko speech banao
        tts = gTTS(response_data["message"], lang="en")
        audio_file_path = os.path.join(
            tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".mp3"
        )
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
if __name__ == "__main__":   # ✅ fixed
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
