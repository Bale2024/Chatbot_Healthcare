import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("healthcare-rag")

# Gemini LLM
model = genai.GenerativeModel("gemini-2.5-flash")

embed_model = SentenceTransformer("stsb-bert-large") # 1024-dim embeddings 
def get_embedding(text: str) -> list[float]: 
    """ Generate embeddings for a given text using SentenceTransformer. 
         Returns a list of floats suitable for insertion into Pinecone. """ 
    embedding = embed_model.encode(text) # returns a NumPy array 
    return embedding.tolist()    

def retrieve_context(query):
    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    contexts = [match["metadata"]["text"] for match in results["matches"]]
    return "\n".join(contexts)

def generate_answer(query, context):

    prompt = f"""
You are a healthcare assistant.

Use only the context below to answer.
If unsure, say you don't know.

Context:
{context}

Question:
{query}

Answer clearly:
"""

    response = model.generate_content(prompt)
    return response.text


# -------- STREAMLIT UI -------- #

st.title("Healthcare Chatbot")

user_query = st.text_input("Ask a healthcare question:")

if user_query:

    with st.spinner("Retrieving knowledge..."):
        context = retrieve_context(user_query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(user_query, context)

    st.markdown("### 🩺 Answer")
    st.balloons()
    st.write(answer)
    st.write("Note: Please ask next healthcare question in the field above.")
    
