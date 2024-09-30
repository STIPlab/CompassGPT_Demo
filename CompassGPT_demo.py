import numpy as np
import pandas as pd
import openai
import tiktoken
import time
from typing import List, Tuple, Dict
import streamlit as st

# Streamlit configuration
st.title("CompassGPT Demo")
st.write("This Streamlit interface is designed to enhance the accessibility of CompassGPT. Please note that this is in a testing stage and for internal use only")

# User inputs for authentication
authenticator = st.text_input("Enter the authentication password", type="password")

# Set your desired password here
AUTH_PASSWORD = "stipdatalab1961"

if authenticator != AUTH_PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# Get the OpenAI API key from secrets
api_key = st.secrets["openai_api_key"]

# Set the API key
openai.api_key = api_key

# Define constants
COMPLETIONS_MODEL = "gpt-4o-mini-2024-07-18"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 11800
SEPARATOR = "\n* "
ENCODING = "cl100k_base"
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

# Define functions
def get_embedding(text: str, model: str = EMBEDDING_MODEL, sleep_time: float = 0.5) -> List[float]:
    response = openai.Embedding.create(model=model, input=[text])
    time.sleep(sleep_time)
    return response['data'][0]['embedding']

def load_embeddings(fname_list: List[str], countries: List[str] = None) -> Dict[Tuple[str, str], List[float]]:
    df_list = [pd.read_csv(fname) for fname in fname_list]
    df = pd.concat(df_list, ignore_index=True)
    if countries:
        df = df[df["title"].str.contains("|".join(countries))]
    max_dim = max([int(c) for c in df.columns if c not in ["title", "heading"]])
    return {(r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()}

def filter_embeddings_by_countries(embeddings: Dict[Tuple[str, str], List[float]], countries: List[str]) -> Dict[Tuple[str, str], List[float]]:
    return {key: value for key, value in embeddings.items() if any(country in key[0] for country in countries)}

def vector_similarity(x: List[float], y: List[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: Dict) -> List[Tuple[float, Tuple[str, str]]]:
    query_embedding = get_embedding(query)
    return sorted([(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()], reverse=True)

def construct_prompt(question: str, context_embeddings: Dict, df: pd.DataFrame, instruction: str, show_policies: bool) -> str:
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    chosen_sections, chosen_sections_len, chosen_sections_indexes = [], 0, []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    if show_policies:
        st.write(f"Selected {len(chosen_sections)} policy initiatives:")
        st.write("\n".join(chosen_sections_indexes))

    return instruction + "".join(chosen_sections) + "\n###\n Q: " + question + "\n"

def answer_query_with_context(query: str, df: pd.DataFrame, document_embeddings: Dict, instruction: str) -> str:
    prompt = construct_prompt(query, document_embeddings, df, instruction, show_policies=True)
    response = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content']

# Load the embeddings from local files
file_paths = [
    'document_embeddings_part1.csv',
    'document_embeddings_part2.csv',
    'document_embeddings_part3.csv'
]
document_embeddings = load_embeddings(file_paths)

countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Korea', 'Latvia', 'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Türkiye', 'United Kingdom', 'United States']
filtered_embeddings = filter_embeddings_by_countries(document_embeddings, countries)

url = 'https://stiplab.github.io/CompassGPT/2023-STIP_Compass-for-embedding.csv'
df = pd.read_csv(url, encoding='UTF-8-SIG').set_index(["title", "heading"])

# Streamlit interface for querying
query = st.text_input("Enter your question")

if api_key and query:
    openai.api_key = api_key
    instruction = "Use the context to answer the question."
    answer = answer_query_with_context(query, df, filtered_embeddings, instruction)
    st.write("Answer:")
    st.write(answer)
