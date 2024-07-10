import streamlit as st
from llm_models import chat_with_chatgpt, chat_with_gemini, chat_with_claude
from vector_search import vector_search, add_chunks_to_index
from web_search import google_search
import fitz  # PyMuPDF

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Multipurpose Personal Chatbot with LLM and Vector DB Support")
st.sidebar.title("Select LLM Model")
model = st.sidebar.radio("Choose a model", ('ChatGPT', 'Gemini', 'Claude'))

user_input = st.text_input("You: ", "Hello, how are you?")

if st.button("Send"):
    # Append user input to history
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Generate response based on selected model and conversation history
    if model == 'ChatGPT':
        response = chat_with_chatgpt(st.session_state.history)
    elif model == 'Gemini':
        response = chat_with_gemini(st.session_state.history)
    elif model == 'Claude':
        response = chat_with_claude(st.session_state.history)
    
    # Append bot response to history
    st.session_state.history.append({"role": "assistant", "content": response})

    # Search vector database and web
    vector_responses = vector_search(user_input)
    web_response = google_search(user_input)

    # Combine responses (example logic)
    final_response = f"{response}\n\nVector DB:\n{' '.join(vector_responses)}\n\nWeb Search:\n{web_response}"

    st.session_state.history[-1]["content"] = final_response  # Update the last bot response with combined response

for message in st.session_state.history:
    role = "User" if message["role"] == "user" else "Bot"
    st.write(f"{role}: {message['content']}")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    add_chunks_to_index(pdf_text)
    st.write("PDF content indexed successfully.")
