
import asyncio
from itertools import chain
from collections import OrderedDict
from typing import List, Tuple
import logging

from llama_index.retrievers import (BaseRetriever,
                                    BM25Retriever)
from llama_index.postprocessor import LLMRerank
from llama_index.schema import QueryBundle
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from llama_index.memory import ChatMemoryBuffer
from vectorstore.vector_store import DataIngestion, VectorStore
from prompts.query_rewriting_prompt import query_rewrite_prompt
from config.core import config
from utils.utils_data import pp_retriever_query_results

logger =logging.getLogger(__name__)
logger.setLevel(logging.INFO)
                    

class BM25Retrvr(DataIngestion):
    '''Class consituting BM25 retriever '''
    def __init__(self, documents:dict):
        self.docs = documents

    def bm25_retriever(self, **kwargs):
        '''Retrieve data from vector store using keywords'''
      
        nodes = super().prepare_documents(self.docs)
        return BM25Retriever.from_defaults(nodes=nodes, **kwargs)


class HybridRetriever(BaseRetriever):
    ''' Combined BM25 retriver and vector retriever'''
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever =vector_retriever
        self.bm25_retriever =bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes =self.bm25_retriever.retrieve(query,**kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        #combine the two lists of nodes
        all_nodes =[]
        node_ids =set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

class ResultRetriever:

    async def _get_retrieved_nodes(self,
                            query,
                            query_results,
                            index,
                            retriever_type, 
                            similarity_top_k, 
                            rerank_top_n,
                            with_reranker):
        
        query_bundle = QueryBundle(query)

        BM25 = BM25Retrvr(query_results)
        bm25_retriever = BM25.bm25_retriever(similarity_top_k=similarity_top_k)
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        if retriever_type =='bm25':
            retrieved_nodes = bm25_retriever.retrieve(query_bundle)
        elif retriever_type == 'hybrid':
            retrieved_nodes = hybrid_retriever.retrieve(query_bundle)
        else:
            retrieved_nodes = vector_retriever.retrieve(query_bundle)

        if with_reranker:
            reranker = LLMRerank(choice_batch_size=2,
                                top_n=rerank_top_n,)
            retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, 
                                                         query_bundle)
        
        return retrieved_nodes
    
    def _post_processor(self, nodes: list =None):
        if nodes is None:
            raise ValueError("Node list cannot be empty for post processing")
        
        processed_output =[]
        for node in nodes:
            url = node.metadata['url']
            text = node.text
            score = node.score
            processed_output.append((text, url, score))

        return processed_output

    async def retrieve_n_process_nodes(self,
                                       query, 
                                       query_results, 
                                       index,
                                       retriever_type ='vector',
                                       similarity_top_k =20,
                                       rerank_top_n =3,
                                       with_reranker=False):
        retrieved_nodes =  await self._get_retrieved_nodes(query,
                            query_results,
                            index,
                            retriever_type,
                            similarity_top_k,
                            rerank_top_n, 
                            with_reranker)
        pced_output = self._post_processor(retrieved_nodes)
        return pced_output
    
    async def async_retrieve_n_process_nodes(self, 
                                            queries, 
                                            query_results, 
                                            index, 
                                            retriever_type ='vector', 
                                            similarity_top_k =20, 
                                            rerank_top_n =3,
                                            with_reranker=False):
            logging.info("Using {retriever_type} retriever to retrieve documents")
            all_pced_output = await asyncio.gather(*[self.retrieve_n_process_nodes(query,
                                                                      query_results,
                                                                      index,
                                                                      retriever_type,
                                                                      similarity_top_k,
                                                                      rerank_top_n,
                                                                      with_reranker) for query in queries])
            
            combined_output =list(chain.from_iterable(all_pced_output))
            dedup_output = self._dedup_retrieved_results(combined_output)
            return dedup_output

    def _dedup_retrieved_results(self, results:List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        'Remove duplicate entries from retrieved results'
        
        if not isinstance(results, list):
            raise TypeError(f"Expecting type 'list' as input but received '{type(results)}' type")
        
        # remove duplicates by 'URL' field
        seen_values = set()
        dedup_results = [item for item in results if not (item[1] in seen_values or seen_values.add(item[0]))]
        logging.info(f"No of records dropped: {len(results) -len(dedup_results)}")

        return dedup_results

        
def retrieval_query_rewriting(queries):
    '''Method to rewrite and enrich the original search queries for retrieval purpose'''
    
    llm = OpenAI(model= config.model, temperature=0.1)
    chain = {"query_str": RunnablePassthrough()} | query_rewrite_prompt | llm | StrOutputParser()
    batch_input = [{'query_str': query} for query in queries]
    responses= chain.batch(batch_input, config={"max_concurrency": 5})
    pp_results =pp_retriever_query_results(responses)
    return pp_results

def chat_engine(index):
    print("Welcome to the chat engine!")
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    while True:
        user_input = input("You: ")

        # Check if the user wants to end the chat
        if user_input.lower() in ["exit", "end", "quit"]:
            print("Goodbye! Chat ended.")
            break

        chat_engine =index.as_chat_engine(chat_mode ="best",
                                           memory=memory,
                                           system_prompt=("You are a chatbot, able to have normal interactions, as well as give"
                                                          " factual information based on retrieved context information "), 
                                           verbose =True)
        response =chat_engine.chat(user_input)

        print(f"Chatbot: {response}")