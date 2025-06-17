import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os


st.set_page_config(page_title="Document Genie",layout="wide")

st.markdown("""
Docuemnt Genie: Get instant insights from your documents\n
            How it works??\n
Follow these steps to get started:\n
            1. Enter your Google Generative AI API key in the sidebar.\n
        2. Upload a PDF document using the file uploader.\n
        3. Ask questions about the content of the document.

""")

api_key = st.text_input("Enter your Google Generative AI API key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""Answer the question as detailed as possible, from the provided context, make sure to provide all the details, dont provide wrong answer
    context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ",response["output_text"])


#UI

def main():
    st.header("Document Genie")

    user_question = st.text_input("Ask a question about the document:", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF documents", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Documents processed and vector store created successfully!")

if __name__ == "__main__":
    main()
            