from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
)

chat = ChatHuggingFace(llm=llm)

res = chat.invoke("What is the capital of France?")
print(res.content)
