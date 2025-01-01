import streamlit as st
import os
import openai
import sys

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    """children start kindergarten at the age of 6""",
]
st.write(texts)
smalldb = Chroma.from_texts(texts, embedding=embedding)
question = st.text_input("Frage:", "when do children start kindergarten?")
result = smalldb.similarity_search(question, k=3)
unique_items = []
[unique_items.append(item) for item in result if item not in unique_items]
st.write(unique_items)