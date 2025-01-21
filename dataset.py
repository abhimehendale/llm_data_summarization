from langsmith import wrappers,Client
from pydantic import BaseModel,Field
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

client = Client()
openai_client = wrappers.wrap_openai(OpenAI())


examples = [

  (
    "which conuntry is Mount Kilimanjaro located in?",
    "Mount Kilimanjaro is located in Tanzania ",
  ),
  (
    "What is earth's lowest point",
    "Earth's lowest point is Dead sea",
  )
]

inputs = [{"question": input_prompt} for input_prompt,_ in examples]
outputs = [{"answer": output_prompt} for _,output_prompt in examples]


dataset = client.create_dataset(
  dataset_name="Sample Dataset", description="First test dataset"
)

client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)






