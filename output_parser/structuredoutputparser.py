from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema




load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation")

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = 'Fact1', description = 'Fact1 about the topic'),
    ResponseSchema(name ='Fact2', description = 'Fact2 about the topic'),
    ResponseSchema(name = 'Fact3', description = 'Fact3 about the topic')
]

parser = StructuredOutputParser.from_response_schema(schema)

template = PromptTemplate(
    template = 'Give me 3 facts about the {topic} \n {format_instruction}',
    input_variable = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic' : 'Moon'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)