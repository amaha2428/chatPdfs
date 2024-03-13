from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

load_dotenv()

# api_key = os.getenv('GOOGLE_API_KEY')
api_key = st.secrets['GOOGLE_API_KEY']

llm = GooglePalm(google_api_key=api_key, temperature=0.5)

def get_pdf_text(pdf_docs):

    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f'Error {str(e)}')

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_db(chunk):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunk, embedding=embeddings)
    return vectorstore
    

def user_question(vectorstore):
    # llm = ChatOpenAI()

    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm =llm,
    #     retriever=vectorstore.as_retriever()
    # )

    # return conversation_chain
    retriever = vectorstore.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "Your request is beyound my scope of assessment." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever= retriever,
        input_key= "query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain




