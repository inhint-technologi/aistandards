from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
#import pinecone
import openai
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
#from langchain.llms import openai
import streamlit as st 
from langchain.schema import Document
from langchain_community.document_transformers import DoctranTextTranslator
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    
    import pinecone

    pinecone.init(
        api_key =PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name ='aichatstandard'
    )
    return doc_db

# note: these 2 lines are not strictly necessary, just testing client connection
#pc_index = pinecone.Index('aichatstandard')
#pc_index.describe_index_stats() 

llm = ChatOpenAI()
doc_db = embedding_db()
#result =""

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
    st.title("Question and Answering App powered by LLM and Pinecone")
    text_input = st.text_input("Ask your Query...")
    if st.button("Ask Query"):
        if len(text_input)>0:
            st.info("Your query: " + text_input)
            retriever = doc_db.as_retriever()
            
            # translat from Norwegian to English
            #qa_translator = DoctranTextTranslator(language="english") 
            #documents = [Document(page_content=text_input)]
            #result2 = qa_translator.transform_documents(documents)
            result3 = translate_text(text_input, "english") 
    
            answer = retrieval_answer(result3, retriever)
            st.success(answer)
if __name__ == "__main__":
    main()