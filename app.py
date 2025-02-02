import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


import streamlit as st
from PyPDF2 import PdfReader
from typing import List

# Initialize configuration FIRST
st.set_page_config(
    page_title="Chat PDF",
    page_icon="💁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (must come after set_page_config)
st.markdown("""
    <style>
        .answer-box { 
            padding: 20px; 
            background: #f0f2f6; 
            border-radius: 10px; 
            margin-top: 10px;
            border-left: 4px solid #1e3d6d;
        }
        .question-text { 
            color: #1e3d6d; 
            font-weight: 500;
            margin-bottom: 10px;
        }
        .warning { 
            color: #ff4b4b; 
            font-weight: 500;
            padding: 10px;
            border-radius: 5px;
            background: #fff3f3;
        }
        .stButton>button {
            background-color: #1e3d6d;
            color: white;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []
    if 'history' not in st.session_state:
        st.session_state.history = []

def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        if pdf.type != "application/pdf":
            continue
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text: str) -> List[str]:
    return [text[i:i+1000] for i in range(0, len(text), 1000)]

def get_vector_store(text_chunks):
    # Add your vector store implementation here
    pass

def get_conversational_chain():
    # Simplified model interaction
    class FakeModel:
        def __call__(self, prompt, temperature=0.3):
            return f"Sample response to: {prompt} (Temperature: {temperature})"
    
    prompt_template = "Context: {context}\n\nQuestion: {question}\nAnswer:"
    return FakeModel(), prompt_template

def main():
    initialize_session_state()
    
    st.header("📚 Chat with PDF using Gemini 💁")
    
    # Sidebar Section
    with st.sidebar:
        st.title("Menu")
        
        # File Upload Section
        with st.expander("📁 Upload PDFs", expanded=True):
            pdf_docs = st.file_uploader(
                "Upload your PDF Files", 
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if pdf_docs:
                for pdf in pdf_docs:
                    if pdf.type != "application/pdf":
                        st.warning(f"⚠️ File {pdf.name} is not a PDF!", icon="⚠️")

            process_button = st.button("Submit & Process")
        
        # Processing Section
        if process_button and pdf_docs:
            with st.spinner("⏳ Processing PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.text_chunks = get_text_chunks(raw_text)
                    get_vector_store(st.session_state.text_chunks)
                    st.session_state.processed = True
                    st.success("✅ Processing Complete!")
                    st.session_state.history = []
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
        elif process_button and not pdf_docs:
            st.warning("⚠️ Please upload PDF files first!")

        # Question Section
        st.markdown("---")
        st.title("💬 Ask Questions")
        user_question = st.text_input(
            "Ask a Question from the PDF Files",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        
        # Settings Section
        st.markdown("---")
        with st.expander("⚙️ Settings"):
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3,
                help="Control response creativity"
            )
        
        if st.button("🔄 Reset Conversation"):
            st.session_state.clear()
            st.rerun()

    # Main Content Section
    col1, col2 = st.columns([3, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/711/711245.png", width=150)

    # Chat Interface
    if user_question:
        if not st.session_state.processed:
            st.markdown("<p class='warning'>❗ Please upload and process PDFs first!</p>", 
                       unsafe_allow_html=True)
        else:
            with st.spinner("🧠 Generating Answer..."):
                try:
                    model, prompt_template = get_conversational_chain()
                    context = " ".join(st.session_state.text_chunks)
                    prompt = prompt_template.format(context=context, question=user_question)
                    response = model(prompt, temperature=temperature)
                    
                    st.session_state.history.append({
                        "question": user_question,
                        "answer": response
                    })

                    with st.container():
                        st.markdown(f"<p class='question-text'>🗨️ Your Question:<br>{user_question}</p>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"<div class='answer-box'>📖 Answer:<br>{response}</div>", 
                                   unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

    # Chat History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Chat History")
        for entry in reversed(st.session_state.history):
            with st.expander(f"Q: {entry['question'][:50]}..."):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Answer:** {entry['answer']}")

    # Help Section
    with st.sidebar.expander("ℹ️ How to Use"):
        st.markdown("""
            1. **Upload PDF files**
            2. Click **Submit & Process**
            3. Ask questions
            4. Adjust **temperature**
            5. Use **Reset Conversation** to start over
        """)

if __name__ == "__main__":import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="💁", layout="wide")
    st.header("Chat with PDF using Gemini💁")

    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success("Processing Done")

    st.sidebar.markdown("---")
    st.sidebar.title("Ask Questions:")
    user_question = st.sidebar.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Generating Answer..."):
            model, prompt_template = get_conversational_chain()
            context = " ".join(text_chunks)  # Assuming text_chunks is available globally or passed appropriately
            prompt = prompt_template.format(context=context, question=user_question)
            response = model(prompt)
            st.write("### Question:")
            st.write(user_question)
            st.write("### Answer:")
            st.write(response)

    st.sidebar.markdown("---")
    st.sidebar.title("Settings:")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
    st.sidebar.write("Adjust the model's creativity with the temperature slider.")

if __name__ == "__main__":
    main()
    main()