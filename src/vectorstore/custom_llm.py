from llama_index import ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LangChainLLM, OpenAI
from langchain.llms.base import LLM
from transformers import pipeline
from llama_index import PromptHelper
import torch
import os
from pathlib import Path
from dotenv import load_dotenv
from config.core import config
from config import core

env_path = Path(__file__).parents[2] / '.env'
load_dotenv(env_path)

embedding_model_dir = core.MODEL_DIR / 'embedding_model'
embedding_model = HuggingFaceEmbedding(model_name = config.embed_model_name, 
                                   cache_folder = embedding_model_dir)

prompt_helper =PromptHelper(context_window=config.context_window,
                        num_output=config.num_output,
                        chunk_overlap_ratio=config.chunk_overlap_ratio,
                        separator=config.separator)

class LocalLLM(LLM):
    model_name = config.generative_model_name
    pipeline = pipeline("text-generation",
                        model= model_name,
                        model_kwargs= {"torch_dtype" :torch.bfloat16})
 
    def _call(self, prompt, stop =None):
        print(prompt)
        response = self.pipeline(prompt, max_new_tokens = config.max_tokens)[0]["generated_text"]
        print(response)
        return response
    
    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}
    
    @property
    def _llm_type(self):
        return "custom"
    
openai_llm = OpenAI(model=config.model, 
                    temperature= config.temperature, 
                    max_tokens= config.max_tokens,
                    api_key=os.getenv('OPENAI_API_KEY'))

service_context = ServiceContext.from_defaults(
        llm =openai_llm if config.llm.lower() =="openai" else LangChainLLM(LocalLLM()),
        prompt_helper=prompt_helper,
        embed_model =embedding_model)

