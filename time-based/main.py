import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
import os
import asyncio
from datetime import datetime

# Page settings
st.set_page_config(page_title="Time Based Greeting Agent", page_icon="🕰️")

# Custom CSS for styling
st.markdown("""
<style>
.stButton>button {
    background-color: #6a0dad;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 16px;
    border: none;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #4b0082;
}

h1, h2, h3 {
    color: #6a0dad;
}
</style>
""", unsafe_allow_html=True)

# Load .env and get API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Please set the GEMINI_API_KEY in the .env file.")
    st.stop()

# Initialize Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Setup model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Streamlit App
st.title("🕰️ Time-Based Greeting Agent")
st.subheader("Get a warm greeting based on your local time 🌞🌙")

user_input = st.text_input("Enter your name:")

def get_greeting() -> str:
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good afternoon"
    elif 18 <= current_hour < 20:
        return "Good evening"
    else:
        return "Good night"

async def run_agent(name):
    time_slot = get_greeting()
    agent = Agent(
        name="Time Based Greeting Agent",
        instructions=f"You are a polite AI agent. Greet {name} based on the time of day ({time_slot}). Mix Urdu and English. Add a warm tone and emojis.",
        model=model
    )
    result = await Runner.run(agent, f"Give a {time_slot} greeting to {name}", run_config=config)
    return result.final_output

if st.button("Generate Greeting"):
    if user_input:
        with st.spinner("Generating your greeting..."):
            try:
                greeting = asyncio.run(run_agent(user_input))
                st.success("Here is your time-based greeting:")
                st.markdown(f"### 📨 {greeting}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter your name.")
