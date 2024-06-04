import os
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms.ctransformers import CTransformers
from  langchain_community.llms.llamacpp import LlamaCpp
import streamlit as st 

load_dotenv("./example.env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

model_name="sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

load_vector_store = Chroma(persist_directory="stores/rag_data", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":4})

# model="model\mistral-7b-v0.1.Q2_K.gguf"
# llm=CTransformers(model=model,
#                 config={'max_new_tokens':512,
#                           'temperature':0.01,
#                           'context_length':1024})

# model = r"/home/rudraksh/RAG_DEMO/Lightning Talk 1/LLM model/mistral-7b-v0.1.Q2_K.gguf"
model = r"/home/rudraksh/RAG_DEMO/Lightning Talk 1/LLM model/Phi-3-mini-4k-instruct-q4.gguf"
# model = r"/home/rudraksh/RAG_DEMO/Lightning Talk 1/LLM model/gemma-2b-it.Q2_K.gguf"

llm = LlamaCpp(model_path=model,
               temperature=0.1,
               max_tokens=2000,
               verbose=False,
               n_ctx=2048)
print(llm)

chain_type_kwargs = {"prompt": prompt}
def qa_chain():
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
    )
    return qa
qa = qa_chain()


def main():
    st.set_page_config(page_title="RAG Demo", page_icon=":books:")
    
    st.title("Hello RAG 👋")
    text_query = st.text_area("Ask your Question")
    generate_response_btn = st.button("Run RAG")
    st.subheader("Response")
    
    if generate_response_btn and text_query:
        start_time=time.time()
        with st.spinner("Generating response..."):
            text_response = qa(text_query)
            end_time=time.time()
            st.write(f"Time Taken:{round((end_time-start_time)/60,2)} Minutes")
            if text_response:
                st.write(text_response)
                st.success("Response generated!")
            else:
                st.error("Failed to generate response.")

if __name__ == "__main__":
    main()