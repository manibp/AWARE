import time
import sys

import streamlit as st
from streamlit_chat import message
from collections import namedtuple
import logging
import asyncio
import pandas as pd
from dotenv import load_dotenv

from translation.query_translation import abatch_query_translation    
from utils.utils_data import load_lang_dict, load_country_dict, load_ctry_lang_mapping
from search.search_utils import (prepare_seach_queries,
                                 langs_2_search)
from search.web_search import ApifyGoogleSearch
from vectorstore.vector_store import VectorStore, DataIngestion
from retrieval.retrieval_n_generation import ResultRetriever, retrieval_query_rewriting

logger =logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

TranslationResult = namedtuple('TranslationResult', ["languages", "queries", "tr_results"])
SearchResults = namedtuple('SearchResults', ["queries", "country", "sr_results"])

                
# Streamlit app
if __name__ == "__main__": 

    st.set_page_config(page_title="AWARE: Adverse Media Assesment and Reporting Engine", layout="wide")
    st.title("Adverse Media Assesment and Reporting Engine")    

    @st.cache_data
    def st_load_lang_dict():
        lang_dict = load_lang_dict()
        inv_lang_dict = {v: k for k, v in lang_dict.items()}
        return lang_dict, inv_lang_dict

    lang_dict, inv_lang_dict = st_load_lang_dict()
    lang_list = list(lang_dict.keys())
    default_lang_idx = lang_list.index('English')

    @st.cache_data
    def st_load_country_dict():
        country_dict = load_country_dict()
        return country_dict
    
    country_list = list(st_load_country_dict().keys())
    default_ctry_idx = country_list.index('United States')

    @st.cache_data
    def st_load_ctry_lang_mapping():
        ctry_lang_dict = load_ctry_lang_mapping()
        return ctry_lang_dict
    
    ctry_lang_map_dict = st_load_ctry_lang_mapping()
    is_existing_session = True

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 25px;'<span>&#9881;</span> Settings </h1>", unsafe_allow_html=True)
        st.markdown(" **Web Search Configuration:**  ", unsafe_allow_html=False,)
        inp_country =st.selectbox("Country", country_list, index=default_ctry_idx, help="Specifies the country of the IP address used for the search and the Google Search domain (e.g. google.es for Spain). By default, the actor uses United States (google.com).")
        inp_language =st.multiselect("Language", lang_list, default=['English'], help= "Language for the search results, which is passed to Google Search as the hl URL query parameter. Only set this if you want to use a non-default language for the selected country.")
        st.markdown(" **Similarity Search Configuration:**  ", unsafe_allow_html=False, help= "Only returns the web search results that pass the criteria")
        score_thresh = st.slider("Similarity Score Threshold", 0.0, 1.0, 0.6, step=0.05)   
        st.sidebar.markdown("LLM used in this app: Sentence Transformer (all-MiniLM-L6-v2)")
        
    logging.info(f"Country Code : {inp_country }, Language : {inp_language}")# , Index Status: {index_status}")
    col1,col2 = st.columns(2)
    inp_name =col1.text_input(label='Individual Name', placeholder='Name of the individual without quotes to search. Ex: Donald Trump ',
                                help= 'If specified, performs web search based on pre-defined set of queries')
    inp_location = col2.text_input(label='Location (Optional)', placeholder='Location of the individual',
                                    help= 'If specified, incorporates location into pre-deined queries')
    inp_additional_text = st.text_input(label='Additional keywords (Optional)', placeholder='Additional keywords. Ex: scandals, drug mafia ..  ')
    st.markdown(" **OR** ", unsafe_allow_html=False)
    custom_query = st.text_area("Custom Search Query", value= None, placeholder="Performs web search based on user specified query!!",
                                help= "If specified together with Individual name, search is performed with both user specified query and pre-defined queries, otherwise performs search only with user specified query ")
    
    if st.button("Search"): 
        is_existing_session = False

        if not inp_name and not custom_query:
            st.warning("Please provide Search Query or Entity name to search.")
        else:
            with st.spinner('Performing Multi Lingual Search...'):
                st.header("Search Results")
                start_time = time.time()

                #Fetch languages to conduct search based on country domain, and choosen languages
                @st.cache_data
                def st_langs_2_search(country, languages):
                    return langs_2_search(country, languages)
                
                search_langs =st_langs_2_search(inp_country, inp_language)

                #Prepare the search queries based on the inputs
                @st.cache_data
                def st_prepare_search_queries(name, custom_search_query, location, add_keywords):
                    return prepare_seach_queries(name,
                                                    custom_search_query,
                                                    location,
                                                    add_keywords)
                search_queries = st_prepare_search_queries(inp_name, custom_query, inp_location, inp_additional_text)
                
                # Query translation
                async def st_abatch_query_translation(search_langs, search_queries):
                    translated_queries = await abatch_query_translation(search_langs,search_queries)
                    return TranslationResult(search_langs, search_queries, translated_queries)
                
                translated_results =asyncio.run(st_abatch_query_translation(search_langs, search_queries))
                logging.info(translated_results.tr_results)
                st.session_state['translated_results'] = translated_results.tr_results

                with st.expander('Search Queries'):
                    languages, queries = zip(*st.session_state['translated_results'])
                    st.write(f"Searching in langugages: {list(set([inv_lang_dict[lang] for lang in languages]))} ")
                    st.write(f"Issued Queries: {[query for query in queries]} ")     

                # Issue Web search and collect results
                async def st_aperform_batch_searches(translated_queries, country):
                    logging.info("First async non cached call ..")
                    AGS = ApifyGoogleSearch()
                    search_responses =  await AGS.aperform_batch_searches(translated_queries, country)
                    return SearchResults(translated_queries, country, search_responses)

                search_results = asyncio.run(st_aperform_batch_searches(translated_results.tr_results, inp_country))

                if not search_results:
                    st.warning("No results found.")
                    sys.exit("No results found")
                else:
                    search_results_df =pd.DataFrame(search_results.sr_results, columns =['title', 'description', 'url'])

                    ## Display Web Search Results
                    with st.expander("Web Search Results"):
                        st.dataframe(search_results_df, 
                                    column_config={"title": "Article Headline", 
                                                    "url": st.column_config.LinkColumn("Source"),
                                                    "description": "Short Summary"},
                                    hide_index=True,
                                    use_container_width =True)
                    logging.info(f"Length of dataframe: {len(search_results_df)}")
                st.session_state['search_results'] = search_results #cache the results


                # Create Vector Store Index
                def st_vstore_index():
                    VS = VectorStore()
                    return VS.create_vstore_index()
                
                vindex = st_vstore_index()

                # Data Ingestion to Vector Store
                def st_data_ingestion(index, results):
                    DI = DataIngestion()
                    vs_index = DI.ingest_data_2_index(results, index)
                    return vs_index
                
                l_vindex = st_data_ingestion(vindex, search_results.sr_results)
                st.session_state['vs_index'] = l_vindex
                
                # Rewrite the original search queries for better retrieval
                rr_search_queries = retrieval_query_rewriting(search_queries)
                logging.info(f"Retrieval queries: {rr_search_queries}")

                #Result retrieval from database
                async def st_retriever(queries, results, index,**kwargs):
                    RR =ResultRetriever()
                    retrieved_responses = await RR.async_retrieve_n_process_nodes(queries, 
                                                                                  results, 
                                                                                  index, 
                                                                                  **kwargs)
                    return retrieved_responses
                
                pre_score_fltrd_results =asyncio.run(st_retriever(rr_search_queries, 
                                                                  search_results.sr_results, 
                                                                  l_vindex, 
                                                                  retriever_type ='bm25',
                                                                  similarity_top_k =100))

                # Semantic Score based filtering
                def semantic_score_filtering(response, score_threshold):
                    return [item for item in response if item[2] > score_threshold]
                
                refined_results = semantic_score_filtering(pre_score_fltrd_results, score_thresh)
                if not refined_results:
                    st.warning("No results found.")
                else:
                    with st.expander("Refined Results"):
                        result_df = pd.DataFrame(refined_results, columns =['description', 'url', 'score'])
                        st.dataframe(result_df, 
                                    column_config={"description": "Article Title & Description", 
                                                    "url": st.column_config.LinkColumn("Source"),
                                                    "score": "Relevance Score"},
                                    hide_index=True,
                                    use_container_width =True)
                st.session_state['refined_results'] = refined_results #cache the results
                    
                logging.info(f" Total Execution time: {time.time() - start_time}")
                stats_dict = {"# Web Results": search_results_df.shape[0], "# Refined Results": 0 if not refined_results else result_df.shape[0]}
                
                # Display Statistics
                with st.container():
                    st.header("Statistics")

                    outer_box_html = '<div style="display: flex; padding: 10px; border-radius: 10px;">'
                    def generate_columns(values):
                        columns_html = ''
                        for value in values:
                            column_html = f'<div style="flex: 1; padding: 10px;  border-radius: 5px; margin-right: 10px; font-weight: bold;">{value}</div>'
                            columns_html += column_html
                        return columns_html
                    outer_box_html += generate_columns(stats_dict.keys())
                    outer_box_html += '</div><div style="display: flex;">'  # Start a new row
                    outer_box_html += generate_columns(stats_dict.values())
                    outer_box_html += '</div></div>'
                    st.markdown(outer_box_html, unsafe_allow_html=True)

    ## Display the results of old session, if no new session initiated
    if st.session_state:
        if is_existing_session:
            with st.expander('Search Queries'):
                languages, queries = zip(*st.session_state['translated_results'])
                st.write(f"Searching in langugages: {list(set([inv_lang_dict[lang] for lang in languages]))} ")
                st.write(f"Issued Queries: {[query for query in queries]} ")   
        
            if not st.session_state['search_results']:
                st.warning("No results found.")
                sys.exit("No results found")
            else:
                search_results_df =pd.DataFrame(st.session_state['search_results'].sr_results, columns =['title', 'description', 'url'])

                ## Display Web Search Results
                with st.expander("Web Search Results"):
                    st.dataframe(search_results_df, 
                                column_config={"title": "Article Headline", 
                                                "url": st.column_config.LinkColumn("Source"),
                                                "description": "Short Summary"},
                                hide_index=True,
                                use_container_width =True)

            if not st.session_state['refined_results']:
                st.warning("No results found.")
            else:
                with st.expander("Refined Results"):
                    result_df = pd.DataFrame(st.session_state['refined_results'], columns =['description', 'url', 'score'])
                    st.dataframe(result_df, 
                                column_config={"description": "Article Title & Description", 
                                                "url": st.column_config.LinkColumn("Source"),
                                                "score": "Relevance Score"},
                                hide_index=True,
                                use_container_width =True)
    
        ## Chat Bot

        from llama_index.memory import ChatMemoryBuffer
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        @st.cache_resource
        def chat_engine(_index, user_input = None, reset=False):

            logger.info("Welcome to the chat engine!")
            chat_engine =_index.as_chat_engine(chat_mode ="condense_plus_context",
                                            memory=memory,
                                            system_prompt=("You are a chatbot, able to have normal interactions, as well as give"
                                                            " factual information based on retrieved context information. "), 
                                            verbose =True)
            if reset:
                chat_engine.reset()
            else:
                if user_input is None:
                    raise Exception("Please ask your question.")
                response =chat_engine.chat(user_input)
                logger.info(f"Chatbot: {response}")
                return response

        query =st.chat_input(placeholder ="Hi, do you have follow up questions?. Ask me here")
        
        st.divider()
        st.subheader("AWARE AI Help :female-detective:")
        with st.container():
            if not is_existing_session and ("user_messages" in st.session_state.keys()):
                del st.session_state["user_messages"]
                del st.session_state["bot_messages"]

            if "user_messages" not in st.session_state:
                st.session_state.user_messages = []

            if "bot_messages" not in st.session_state:
                st.session_state.bot_messages = []

            if query:
                st.session_state.user_messages.append({"role": "user", "content": query})
                # message(query, is_user =True, key= '_user')

                bot_response = chat_engine(st.session_state['vs_index'],query).response
                st.session_state.bot_messages.append({"role": "ai", "content": bot_response})

                # Display chat messages from history on app rerun
                for i in range(len(st.session_state.user_messages)):
                    message(st.session_state['user_messages'][i]["content"], is_user =True, key =str(i) +'_user')
                    message(st.session_state['bot_messages'][i]["content"], key= str(i))

            reset =st.button("Reset Chat")
            if reset:
                chat_engine(st.session_state['vs_index'], reset=True)
                del st.session_state["user_messages"]
                del st.session_state["bot_messages"]


                