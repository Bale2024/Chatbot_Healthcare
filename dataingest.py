import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer   
load_dotenv()

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index parameters
index_name = "healthcare-rag"
dimension = 1024  
metric = "cosine"
cloud = "aws"
region = "us-east-1"

# Check if the index already exists to avoid errors
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud=cloud,
            region=region
        )
    )
    print("Index created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index (optional for serverless, the client handles connection implicitly)
index = pc.Index(index_name)
print(index.describe_index_stats())

# Sample healthcare data
documents = [
    "Diabetes is a chronic condition that affects blood sugar levels.",
    "Hypertension is high blood pressure that increases heart disease risk.",
    "Asthma is a respiratory condition causing breathing difficulty."
]

# Embedding model setup
embed_model = SentenceTransformer("stsb-bert-large") # 1024-dim embeddings 
def get_embedding(text: str) -> list[float]: 
    """ Generate embeddings for a given text using SentenceTransformer. 
         Returns a list of floats suitable for insertion into Pinecone. """ 
    embedding = embed_model.encode(text) # returns a NumPy array 
    return embedding.tolist()    

for i, doc in enumerate(documents):
    embedding = get_embedding(doc)

    index.upsert(
        vectors=[
            {
                "id": str(i),
                "values": embedding,
                "metadata": {"text": doc}
            }
        ]
    )

print("Data uploaded successfully!")

#index = pc.Index("healthcare-rag")
# Configure Gemini
# configure(api_key="AIza***********************xyz")
'''
def embed_text(text):
    model = "models/embedding-001"
    return genai.embed_content(
        model=model,
        content=text
    )["embedding"] '''

'''def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text using Gemini."""
    result = genai.embed_content(
        model="all-MiniLM-L6-v2",
        content=text,
        task_type="retrieval_document"
    )
    embedding = [float(x) for x in result["embedding"]]
    return embedding'''
