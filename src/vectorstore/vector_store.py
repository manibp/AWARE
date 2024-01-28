import logging
import sys
import os
import time

from pathlib import Path
from dotenv import load_dotenv
from llama_index import (VectorStoreIndex, 
                         StorageContext,)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.prompts import PromptTemplate
import pinecone
from llama_index import Document
from llama_index import set_global_service_context

from search.web_search import ApifyGoogleSearch
from config.core import config
from .custom_llm import service_context

set_global_service_context(service_context) # Reduce the overhead of passing context everytime

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

env_path = Path(__file__).parents[2] / '.env'
load_dotenv(env_path)

class VectorStore:
    '''Construct Vector Store and Index '''

    def create_vstore_index(self):
        '''create vector store index'''

        logging.info(" Initiating vector Store")
        pinecone.init(api_key = os.getenv('PINECONE_API_KEY'), environment = 'gcp-starter' )

        # Create a Pinecone index with metadata settings
        try:
            if config.pinecone_index not in pinecone.list_indexes():
                logging.info("Pinecone Vector Index creation in progress ..")
                pinecone.create_index(
                    name=config.pinecone_index,
                    metric='cosine',
                    dimension=config.embed_model_dims,
                    pod_type='p1'
                )
                time.sleep(4)
            else:
                logging.info("Index already exist. Skipping creation of Index")
            
            return pinecone.Index(config.pinecone_index)
        except Exception as e:
            logging.info(e.message, e.args)

    
    def delete_vstore_index(self):
        ''' Delete vector store index'''

        if config.pinecone_index in pinecone.list_indexes():
            logging.info(f"deleting {config.pinecone_index} ..")
            pinecone.delete_index(config.pinecone_index)
            time.sleep(4)

class DataIngestion:

    def prepare_documents(self, raw_docs:dict):  
        ''' Convert input documents to LlamaIndex Documents class'''
        
        if not raw_docs:
            raise ValueError("No documents supplied to store in vector store")

        if not isinstance(raw_docs, dict):
            raise TypeError(f"Expected dictionary as input, but received {type(raw_docs)}")

        logging.info('Generating Documents ..')

        docs =[]
        for title, des, url in zip(raw_docs['title'], raw_docs['description'], raw_docs['url']):
            text = 'Title: ' + title.strip() + '; Content:' + des.strip()
            metadata = {'url': url}
            docs.append(Document(text=text, 
                                 metadata=metadata.copy(), 
                                 excluded_llm_metadata_keys=["url"]))
    
        self.pced_docs =docs

        return docs

    def ingest_data_2_index(self, raw_docs, vsindex):
        ''' Uploading result embedding to vector index'''

        self.prepare_documents(raw_docs)

        storage_context = StorageContext.from_defaults(
            vector_store=PineconeVectorStore(vsindex)
        )
        
        self.index = VectorStoreIndex.from_documents(self.pced_docs,
                                                storage_context=storage_context)
        return self.index


if __name__ == '__main__':
    import asyncio
    country_code ='us'
    lang='en'
    search_query = "Information about India's Aditya L1 mission"
    AGS = ApifyGoogleSearch()

    async def main():
        # print(AGS.prepare_search_input(search_query, country_code, lang))
        sres =await asyncio.gather(AGS.get_search_results(search_query, country_code, lang))
        return sres
        
    search_results =asyncio.run(main())
    print(search_results)

    vs =VectorStore()
    vsindex =vs.create_vstore_index()
    print(vsindex)
    di =DataIngestion()
    l_index =di.ingest_data_2_index(search_results[0], vsindex)
    query_engine =l_index.as_query_engine(similarity_top_k =2)

    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n")

    qa_template = PromptTemplate(template)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})
    res =query_engine.query("tell me about Aditya L1 mission")
    # print(query_engine.get_prompts())
    
    print(res)
    # print(generate_documents(search_results))
    
