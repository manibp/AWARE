import pandas as pd
import openai
import pinecone
import time
import json
import streamlit as st
from apify_client import ApifyClient, ApifyClientAsync
from sentence_transformers import SentenceTransformer
from google.cloud import translate_v2 as translate
from collections import defaultdict
import os
import logging
import asyncio
import cachetools
import random, string
from google.oauth2 import service_account
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
cache = cachetools.LRUCache(maxsize=128)
from dotenv import load_dotenv

load_dotenv()
chunk_size = 100
id =''.join(random.choices(string.ascii_uppercase +string.digits, k=10))
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcs_credentials"])



# API tokens
def time_deco(id): 
    def outer_wrapper(func):      
        def wrapper(*args, **kwargs):
            nonlocal id
            start = time.time()
            result = func(*args, **kwargs)
            time_elapsed =  time.time() - start
            print(f"Time elapsed for {func.__name__} is {time_elapsed}")
            with open('./func_time_log.txt', 'a+') as f:
                f.write(f" Run Id: {id}. Time elapsed for {func.__name__} is {time_elapsed} \n")
            return result
        
        return wrapper
    
    return outer_wrapper

class WebSearchAPI:

    def __init__(self, actor_id, api_token ):
        self.actor_id = actor_id
        self.api_token = api_token

    @time_deco(id)
    def web_search_func(self, input_data):
        # Initialize Apify client
        client = ApifyClient(token=self.api_token)
        logging.info("Running Web search ...")
        # Run the actor
        actor = client.actor(self.actor_id)
        run = actor.call(run_input = input_data )
        results = actor.last_run().dataset().list_items().items

        cleaned_results = dict()
        for collection in results[0]['organicResults']:       
            for key, value in collection.items():
                if key in ('title', 'description', 'url'):
                    if key not in cleaned_results:
                        cleaned_results[key] = [value]
                    else:
                        cleaned_results[key].append(value)

        return cleaned_results
    
    # Input for the actor
    @staticmethod
    def data_input(search_query, country_code ='us', lang ='en' ):
        logging.info(" Preparing data input for web search ")
        return {
            "countryCode": country_code,
            "customDataFunction": "async ({ input, $, request, response, html }) => {\n  return {\n    pageTitle: $('title').text(),\n  };\n};",
            "includeUnfilteredResults": False,
            "languageCode": lang,
            "maxPagesPerQuery": 1,
            "mobileResults": False,
            "queries": search_query,
            "resultsPerPage": 100,
            "saveHtml": False,
            "saveHtmlToKeyValueStore": False
        }
    

class Datastore:

    def __init__(self, database_api_key, embedding_dimension):
        self.database_api_key = database_api_key
        self.embedding_dimension = embedding_dimension
        self.datastore = None
    
    @staticmethod
    def encode_text(text, model):
        return model.encode(text, convert_to_numpy=True)
    
    @time_deco(id)
    def load_data_in_chunks(self, data, chunk_size=100):
        logging.info("Loading data in chunks to datastore")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            logging.info(f"Length of chunk: {len(chunk)}")
            
            # Encode text using Sentence Transformer for this chunk
            chunk["combined"] = ('Title: ' + chunk["title"].str.strip() + '; Content:' + chunk['description'].str.strip())
            chunk["embedding"] =  chunk["combined"].apply(lambda x: self.encode_text(x, model).tolist()) 
            
            # Create a list of dictionaries for upsert, including metadata
            upsert_data = []
            for index, row in chunk.iterrows():
                upsert_data.append(
                    (   
                        str(index),
                        row["embedding"],
                        {
                            "url": row["url"],
                            "description": row["combined"]
                        }  
                    )
                )
            yield upsert_data

    @time_deco(id)
    def create_dstore_index(self, index_name = "search-query-index", delete_index=False ):
        logging.info(" Initiating Data Store")
        # Initialize the Pinecone client
        pinecone.init(api_key = self.database_api_key, environment = 'gcp-starter' )

        # Delete index if exist
        if delete_index and index_name in pinecone.list_indexes():
            logging.info(f"deleting {index_name} ..")
            pinecone.delete_index(index_name)
            time.sleep(2)

        # Create a Pinecone index with metadata settings
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                metric='cosine',
                dimension=self.embedding_dimension
            )
        self.datastore = pinecone.Index(index_name)
        # self.load_data_in_chunks(search_df, chunk_size)
        print(self.datastore.describe_index_stats())
        return self.datastore 

# Translate text based on google transalate api
@time_deco(id)
def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client(credentials=credentials)

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    # print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result["translatedText"]


# Function to perform a semantic search
@time_deco(id)
def perform_semantic_search(db_index, search_query, model, top_k=100, score_thresh =None):
    query_vector = encode_text(search_query, model).tolist()
    response = db_index.query(query_vector, top_k=top_k, include_metadata= True)

    if score_thresh is not None:
        response = [item for item in response['matches'] if item.score > score_thresh]

    refined_response =[]
    if response is not None:
        for item in response:
            refined_response.append(dict(item['metadata'].items()|{'score':item['score']}.items()))
    return refined_response



# Streamlit app
if __name__ == "__main__":

    st.set_page_config(page_title="Adverse Media Assesment Tool", layout="wide")
    st.title("Adverse Media Assesment Tool")
    with open('./language_dict.json', 'r') as f:
        lang_dict =json.loads(f.read())
        inv_lang_dict = {v: k for k, v in lang_dict.items()}
    lang_list = list(lang_dict.keys())
    default_lang_idx = lang_list.index('English')

    with open('./country_dict.json', 'r') as f:
        country_dict =json.loads(f.read())
        country_list = list(country_dict.keys())
        default_ctry_idx = country_list.index('United States')
        logging.info(default_ctry_idx)

    with open('./country_lang_mapping.json', 'r') as f:
        ctry_lang_map_dict = json.loads(f.read())
    
    
    # User input for parameters
    actor_id = os.getenv('actor_id') 
    api_token = os.getenv('api_token') 
    pinecone_api_key = os.getenv('pinecone_api_key') 
    index_name = "search-query-index"
    top_k = 100
    score_thresh = 0.5

    with st.sidebar:
        # st.title("Settings",)
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Tool Configuration</h1>", unsafe_allow_html=True)
        st.markdown(" **Web Search Configuration:**  ", unsafe_allow_html=False,)
        input_country =st.selectbox("Country", country_list, index=default_ctry_idx, help="Specifies the country of the IP address used for the search and the Google Search domain (e.g. google.es for Spain). By default, the actor uses United States (google.com).")
        input_language =st.multiselect("Language", lang_list, default=['English'], help= "Language for the search results, which is passed to Google Search as the hl URL query parameter. Only set this if you want to use a non-default language for the selected country.")
        st.markdown(" **Datastore Configuration:**  ", unsafe_allow_html=False,)
        chunk_size = st.slider("Chunk Size", 0, 200, 20,step=10, help= "Loads results to datastore in chunks of specified size")
        st.markdown(" **Similarity Search Configuration:**  ", unsafe_allow_html=False, help= "Only return the web search results that pass the criteria")
        score_thresh = st.slider("Similarity Score Threshold", 0.0, 1.0, 0.5, step=0.1)
        st.sidebar.markdown("Models used in this app: Sentence Transformer (all-MiniLM-L6-v2)")
        
    logging.info(f"Country Code : {input_country }, Language : {input_language}, Chunk Size : {chunk_size}")# , Index Status: {index_status}")
    search_query = st.text_area("Search Query")

    # def reset():
    #     st.session_state.index_state = False

    if st.button("Search"):
        
        # print("After clicking Button:", index_status)
        if not search_query:
            st.warning("Please provide Search Query.")
        else:
            with st.spinner('Performing Multi Lingual Search...'):
                st.header("Search Results")
                start_time = time.time()
                
                ## Preparing list of languages to perform search
                @st.cache_data
                def search_languages(in_country, in_language):
                    input_ctry_code = country_dict[in_country]
                    ctry_official_lang_list= ctry_lang_map_dict[input_ctry_code]
                    print(ctry_official_lang_list)
                    input_lang_list =['en']
                    for lang in in_language:
                        input_lang_list.append(lang_dict[lang])
                    ctry_official_lang_list.extend(input_lang_list)
                    search_lang_list = list(set(ctry_official_lang_list))

                    return search_lang_list
                
                search_langs = search_languages(input_country, input_language)
                print(search_langs)
        
                # lang_name = []
                # for lang_code in search_langs:
                #     lang_name.append(inv_lang_dict[lang_code])
                input_ctry_code = country_dict[input_country]
                lang_names = ','.join(inv_lang_dict[lang_code] for lang_code in search_langs)

                ## Web Search 
                
                @st.cache_data
                def load_model():
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    return model
                
                @st.cache_data
                def encode_text(text, _model):
                    return _model.encode(text, convert_to_numpy=True)
                
                @time_deco(id)
                @st.cache_data
                def text_translation(languages, search_query):
                    translated_text =[]
                    for lang in languages:
                        # lang_code =lang_dict[lang]
                        translated_text.append((lang, translate_text(lang, search_query)))
                    return translated_text

                translated_query_lang_pair = text_translation(search_langs, search_query)
                logging.info(translated_query_lang_pair)

                with st.expander('Search Queries'):
                    languages, queries = zip(*translated_query_lang_pair)
                    st.write(f"Searching in langugages: {[inv_lang_dict[lang] for lang in languages]} ")
                    st.write(f"Issued Queries: {[query for query in queries]} ")
                    

                @time_deco(id)
                async def perform_web_search(actor_id,api_token,search_query, country_code,language):   
                    searchapi = WebSearchAPI(actor_id, api_token)
                    search_query = search_query +' -translate'
                    input_data = searchapi.data_input(search_query, country_code, language)
                    search_results = searchapi.web_search_func(input_data)
                    logging.info(f"Web search completed for {search_query} in {country_code} domain & {language} language!!")
                    return search_results
                
                @time_deco(id)
                def cached_web_search(actor_id,api_token,search_query, country, language):
                    cache_key = (search_query, country, language)
                    if cache_key in cache:
                        return cache[cache_key]
                    result = perform_web_search(actor_id,api_token,search_query, country,language)
                    cache[cache_key] = result
                    return result
                
                @time_deco(id)
                async def call_search_func_asyncly(country, search_queries):
                    # Create the translation client
                    # client = translate.Client()
                    # client = translate.Client.from_service_account_json(credentials_file)
                    # Create a list to store the tasks
                    tasks = []
                    # Create a task for each target language
                    for query in search_queries:
                        task = asyncio.create_task(cached_web_search(actor_id, api_token, query[1],country, query[0]))
                        tasks.append(task)

                    # Gather and execute the tasks
                    results = await asyncio.gather(*tasks)
                    return results
                
                @time_deco(id)
                @st.cache_data
                def process_search_results(country, search_queries):
                    search_results = asyncio.run(call_search_func_asyncly(country, search_queries))
                    if len(search_results) > 0:
                        search_results_dict = defaultdict(list)
                        # search_results_dict = dict()
                        for item in search_results:
                            for key, value in item.items():
                                search_results_dict[key].extend(value)
                    else:
                        st.warning("No results found.")

                    search_results_df =pd.DataFrame(dict(search_results_dict)).drop_duplicates(subset = ['url'])
                    return search_results_df
                
                search_results_df = process_search_results(input_ctry_code, translated_query_lang_pair)
                model = load_model()
                embedding_dimension = model.get_sentence_embedding_dimension()
                logging.info(f"Embedding Dimension: {embedding_dimension}")

                @time_deco(id)
                @st.cache_resource
                def database_index(pinecone_api_key, embedding_dimension):
                    # logging.info(f"index status: {index_status}")
                    db =Datastore(pinecone_api_key,embedding_dimension)
                    db_index =db.create_dstore_index( index_name)
                    return db, db_index
                
                db, db_index = database_index(pinecone_api_key, embedding_dimension)
                print(db, db_index)
                # if index_status:
                    # index_status_placeholder.selectbox("Reinitialize Index", [False, True], index=0)

                ## Display Web Search Results
                if search_results_df.shape[0]==0:
                    st.warning("No results found.")
                else:
                    with st.expander("Web Search Results"):
                        st.dataframe(search_results_df, 
                                    column_config={"title": "Article Title", 
                                                    "url": st.column_config.LinkColumn("Article URL"),
                                                    "description": "Article Description"},
                                    hide_index=True,
                                    use_container_width =True)
                logging.info(f"Length of dataframe: {len(search_results_df)}")
                logging.info(db_index.describe_index_stats())

                ## Datastore Ingestion
                # db=Datastore(pinecone_api_key,embedding_dimension)
                # @st.cache_data
                # def datastore_ingestion(search_results_df, chunk_size):
                #     async_results = [db_index.upsert(vectors=chunk, async_req=True)
                #         for chunk in db.load_data_in_chunks(search_results_df, chunk_size)]
                #     logging.info("Datastore Ingestion In progress ..")
                #     logging.info([async_result.get() for async_result in async_results])

                @time_deco(id)
                def ingest_data(search_results_df, chunk_size):
                    chunks = [chunk for chunk in db.load_data_in_chunks(search_results_df, chunk_size)]
                    chunks_hash = hash(str(chunks))
                    cache_key = f"{index_name}_{chunks_hash}"
                    logging.info(cache_key)
                    logging.info(st.session_state)
                    
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = True
                        async_results = [db_index.upsert(vectors=chunk, async_req=True) for chunk in chunks]
                        logging.info("Datastore Ingestion In progress ..")
                        logging.info([async_result.get() for async_result in async_results])
                        time.sleep(20)
                
                ingest_data(search_results_df, chunk_size)

                
                # Semantic Search
                # @st.cache_data
                @time_deco(id)
                def wrapper_semantic_search(_db_index, search_query, _model, top_k, score_thresh):
                    return perform_semantic_search(_db_index, search_query, _model, top_k, score_thresh)
                
                match_results = wrapper_semantic_search(db_index, search_query, model, top_k, score_thresh)
                logging.info(len(match_results))
                logging.info(" Script Execution Completed!!")    

                if not match_results:
                    st.warning("No results found.")
                else:
                    with st.expander("Refined Results"):
                        result_df = pd.DataFrame(match_results)
                        st.dataframe(result_df, 
                                    column_config={"description": "Article Title & Content", 
                                                    "url": st.column_config.LinkColumn("Article URL"),
                                                    "score": "Similarity Score"},
                                    hide_index=True,
                                    use_container_width =True)
                    
                logging.info(f" Total Execution time: {time.time() - start_time}")
                stats_dict = {"# Web Results": search_results_df.shape[0], "# Refined Results": 0 if not match_results else result_df.shape[0]}
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

                st.success('Done!')
                # st.dataframe(search_results_df)

    
