from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Summary Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Summary Length", ["Short (1-2 pragraphs)", "Medium (3-5 paragraphs)", "Long (Detailed Explanation)"])

template = load_prompt('template.json')

prompt = template.invoke()

if st.button('summarize'):
    chain = template | model
    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})
    st.write(result.content)
