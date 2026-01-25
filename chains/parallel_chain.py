from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation")

model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation")

model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short question answers from the following text \n {text}',
    input_variables =['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """LangChain is an open-source framework designed to simplify the development of applications that leverage large language models (LLMs). It provides a suite of tools and abstractions to help developers build, manage, and deploy LLM-powered applications more efficiently. LangChain supports various use cases, including chatbots, question-answering systems, content generation, and more. The framework offers features such as prompt management, memory handling, and integration with external data sources, making it easier to create sophisticated AI applications. With its modular design, LangChain allows developers to customize and extend its capabilities to suit their specific needs."""

result = chain.invoke({'text' : text})

print(result)

chain.get_graph().print_ascii()