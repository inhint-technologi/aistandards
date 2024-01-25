from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st 
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob ='**/*.pdf',
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()

    pc= Pinecone(api_key=PINECONE_API_KEY, environment='gcp-starter')
    index_name ='aichatstandard'
    dimension=1536
    metric='cosine'

    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name ='aichatstandard'
    )
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()

def retrieval_answer(query, retriever):
    llm = ChatOpenAI()
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    result = qa.run(query)
    return result

def main():
    st.title("AI Web App Spørsmål og svar drevet av LLM og Pinecone")
    text_input = st.text_input("Formuler ditt spørsmål...")
    if st.button("Send forespørsel"):
        if len(text_input)>0:
            st.info("Ditt spørsmål: " + text_input)
            retriever = doc_db.as_retriever()
            answer = retrieval_answer(text_input, retriever)
            st.success(answer)
if __name__ == "__main__":
    main()