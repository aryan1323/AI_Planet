import streamlit as st
from langchain_groq import ChatGroq


@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = st.secrets["GROQ_API_KEY"]

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",  # Free, fast model on Groq
        temperature=0.2,
        groq_api_key=api_key,
    )
