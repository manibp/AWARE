import time
import logging

from llama_index.llms import OpenAI
import asyncio

from search.web_search import ApifyGoogleSearch
from translation.query_translation import abatch_query_translation
from vectorstore.vector_store import (VectorStore, 
                          DataIngestion)
from retrieval.retrieval_n_generation import (ResultRetriever, 
                                              retrieval_query_rewriting,
                                              chat_engine)
from config.core import config
from config.core import MODEL_DIR

from search.search_utils import (prepare_seach_queries,
                                 langs_2_search)

logger =logging.getLogger(__name__)
logger.setLevel(logging.INFO)

embedding_model_dir = MODEL_DIR / "embedding_model"

## Assembling all components of the App

async def main():

    #Fetch language to conduct search based on country domain, and choosen languages
    search_langs = langs_2_search(config.country, config.languages)

    #Prepare the search queries based on the inputs
    search_queries = prepare_seach_queries(config.name, 
                                        config.custom_search_query, 
                                        config.location, 
                                        config.add_keywords)

    # Translated queries
    translated_queries = await abatch_query_translation(search_langs, 
                                                search_queries)
    # Conduct web search

    AGS = ApifyGoogleSearch()
    search_results =await AGS.aperform_batch_searches(translated_queries, config.country)

    # Store in Vector Database
    VS = VectorStore()
    ini_index =VS.create_vstore_index()

    #Ingest results to Vector Store
    DI = DataIngestion()
    vs_index = DI.ingest_data_2_index(search_results, ini_index)

    # Rewrite the original search queries for better retrieval
    rr_search_queries = retrieval_query_rewriting(search_queries)

    #Retrive relevant results from vector store
    RR =ResultRetriever()
    retrieved_responses = await RR.async_retrieve_n_process_nodes(rr_search_queries, search_results, vs_index)

    return retrieved_responses, vs_index


if __name__ == '__main__':
    # country_code ='in'
    # lang=['en', 'hi']
    # search_query = ["who is vijay mallya? ", "India's Aditya L1 mission"]
    # tled_l_n_q =batch_query_translation(lang,search_query)
    start =time.time()
    # res, index =asyncio.run(main())
    asyncio.run(main())
    # print(res)
    # print(len(res))
    end =time.time()
    # print(res)
    print(f"Total time taken {(end-start)/60}")
    
    # chat_engine(index)