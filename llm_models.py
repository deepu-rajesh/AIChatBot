import openai
import google.generativeai as genai
from anthropic import Anthropic
from anthropic.types.message import Message

# Initialize API keys
openai.api_key = ''
gemini_api_key = ''
claude_api_key = ''

def chat_with_chatgpt(history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

def chat_with_gemini(history):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    messages = [msg['content'] for msg in history]
    response = model.generate_content(messages)
    return response.text.strip()

def chat_with_claude(history):
    client = Anthropic(api_key=claude_api_key)
    messages = [{'role': 'user' if msg['role'] == 'user' else 'assistant', 'content': msg['content']} for msg in history]
    response: Message = client.messages.create(
        max_tokens=150,
        messages=messages,
        model="claude-3-opus-20240229",
        temperature=0.5,
    )
    return response.content[0].text.strip()
