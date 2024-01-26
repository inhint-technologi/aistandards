from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
from pinecone import ServerlessSpec, PodSpec
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

# initialize connection to pinecone (get API key at app.pc.io)
api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
environment = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

# configure client
pc = Pinecone(api_key=api_key)
index_name = 'aichatstandard' 
use_serverless = True


if use_serverless:
    cloud = os.environ.get('PINECONE_CLOUD') or 'PINECONE_CLOUD'
    spec = ServerlessSpec(cloud='aws', region='us-west-2')
else:
    spec = PodSpec(environment=environment)

@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()

    pc = Pinecone(api_key=api_key)
    index_name = 'aichatstandard' 
    use_serverless = True

    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name = index_name
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