from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Use 'from_model_id' to load the model locally
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # <--- Small model safe for CPU
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=100,
        temperature=0.5,
        do_sample=True,
    ),
)

model = ChatHuggingFace(llm=llm)

print("Loading model... this might take 1-2 minutes...")
result = model.invoke("What is the capital of France?")
print(result.content)