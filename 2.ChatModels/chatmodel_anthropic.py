from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model = "claude-3-5-sonnet-20241022", temperature = 0)

result = model.invoke("Write a code in python to calculate the sum of first 10 numbers")
print(result.content)