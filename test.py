import streamlit as st
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the OpenAI embeddings
embedding = OpenAIEmbeddings()

# Sample texts to index
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    """children start kindergarten at the age of 6""",
]

st.write(texts)

# Create the FAISS vector store from the texts
faiss_store = FAISS.from_texts(texts, embedding)

# Input a question for similarity search
question = st.text_input("Frage:", "when do children start kindergarten?")
if question:
    # Perform similarity search
    result = faiss_store.similarity_search(question, k=3)
    
    # Filter for unique items
    unique_items = []
    [unique_items.append(item) for item in result if item not in unique_items]
    
    # Display the results
    st.write(unique_items)
