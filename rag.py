from dotenv import load_dotenv
import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")


##  add stramlit component

st.set_page_config(page_title="gemini QA",layout="centered")
st.title("gemini Q&A rag App")

#load embeddings model

embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

# load vector store
vectordb = Chroma(
    
    embedding_function=embeddings,
    persist_directory="chroma_store"  # recreated here
)

retriever = vectordb.as_retriever()

#prompt template
prompt_template=PromptTemplate(
    input_variables=["context","question"],
    template="""
    you are a helpful assistant. use the following context to answer the question
    context:
    {context}

    question:
    {question}
    answer in a clear and detailed manner
    """
)

# load llm
llm=ChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
    openai_api_key=openai_api_key
)

#load your
parser = StrOutputParser()

#load your chain
chain = prompt_template | llm | parser

## streamlit user question
user_question=st.text_input("Ask your question about gemini")

## process user question

if user_question:
    #step retrieve relevant document
    docs=retriever.get_relevant_documents(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    #step 2 pass through chain
    response = chain.invoke({
    "context": context,
    "question":user_question 
})
    
    st.subheader("Answer")
    st.write(response)

    #optinal.....
    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs):
            st.markdown(f"**chunk{i+1}:**")
            st.markdown(doc.page_content)


