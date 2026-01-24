from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation")

model = ChatHuggingFace(llm = llm)

class Review(BaseModel):
    key_themes : list[str] = Field(description = "Write down all the key themes discussed in the review in a list")
    summary: str = Field(description = "A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description = "Return sentiment of the review either positive, negative or neutral")
    name : Optional[str] = Field(description = "If the review mentions any person's name")
    
structured_model = model.with_structured_output(Review)

result= structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this""")

print(result.name)