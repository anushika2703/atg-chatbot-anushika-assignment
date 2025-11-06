🧠 ATG Chatbot Assignment
📜 Description

A local command-line AI chatbot powered by the TinyLlama model (Hugging Face).
It maintains conversational memory and simulates intelligent dialogue — showcasing how LLMs can be adapted locally for lightweight, offline chatbot experiences.

🧩 Features

🗣️ Conversational context memory (tracks past responses)

⚙️ Modular design — separate memory, model loading, and interface components

🚀 Runs completely offline with TinyLlama for privacy and speed

💬 CLI interface for fast, interactive testing of local LLM responses

📂 Files

chat_memory.py → Manages conversation history and context

model_loader.py → Loads TinyLlama model and tokenizer

interface.py → Handles command-line conversation flow

requirements.txt → Project dependencies

🧪 Run Locally
pip install -r requirements.txt  
python interface.py  

🌟 Future Improvements

Add web-based interface (Flask/Streamlit)

Integrate prompt templates for structured dialogue

Connect to APIs for extended automation
