from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm = llm)

passthrough = RunnablePassthrough()

prompt1 = PromptTemplate(
    template = "Write a detailed report about {topic}",
    input_variables = ['topic'])

prompt2 = PromptTemplate(
    template = "Summarize the following text \n {text}",
    input_variables = ['text']
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic' : 'Climate Change'})

print(result)