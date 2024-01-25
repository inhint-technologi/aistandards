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

def translate_text(text, target_language):
    prompt = f"Translate the following Norwegian text to {target_language}: {text}"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        #messages=[{"role": "system", "content": "You are a helpful assistant that translates Norwegian to English."},
        #          {"role": "user", "content": f"Translate the following Norwegian text to {target_language}: {text}"}],
        max_tokens=150,
        temperature=0.5,
        n=1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    translated_text = response.choices[0].text.strip()
    return translated_text

def retrieval_answer(query, retriever):
    llm = ChatOpenAI()
    #llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=1000)

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
                      
            result3 = translate_text(text_input, "english") 
    
            answer = retrieval_answer(text_input, retriever)
            st.success(answer)
if __name__ == "__main__":
    main()