import logging
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from apify_client import ApifyClient, ApifyClientAsync

from utils.utils_data import load_country_dict, merge_dicts

env_path = Path(__file__).parents[2] / '.env'
load_dotenv(env_path)

class ApifyGoogleSearch:

    def __init__(self, actor_id = None, api_token =None ):
        self._actor_id = actor_id or os.getenv('APIFY_ACTOR_ID')
        self._apify_client = ApifyClient(token=api_token or os.getenv('APIFY_API_KEY'))
        self._apify_client_async = ApifyClientAsync(token=api_token or os.getenv('APIFY_API_KEY'))

        if self._actor_id is None:
            raise ValueError("'actor_id' cannot be empty")

    def perform_batch_searches(self, search_lang_query_tuple, country ):
        try:
            country_dict = load_country_dict()
            ctry_code =country_dict[country]
        except KeyError:
            raise KeyError(f'No country to country code mapping found for {country}')
        
        search_results =[]
        for s_lang, s_query in search_lang_query_tuple:
            search_results.append(self.get_search_results(s_query, ctry_code, s_lang))
        return search_results
    
    async def aperform_batch_searches(self, search_lang_query_tuple, country ):
        try:
            country_dict = load_country_dict()
            ctry_code =country_dict[country]
        except KeyError:
            raise KeyError(f'No country to country code mapping found for {country}')
        
        search_results =await asyncio.gather(*[self.aget_search_results(s_query, ctry_code, s_lang) 
                         for s_lang, s_query in search_lang_query_tuple])
        return merge_dicts(*search_results)

    def get_search_results(self, search_query, country_code, lang ):
    
        if not search_query:
            raise ValueError("No value specified for search query")
        
        logging.info("Running Web search ...")
        search_input = self.prepare_search_input(search_query, country_code,lang)
        actor = self._apify_client.actor(self._actor_id)
        actor_call =actor.call(run_input = search_input)
        search_results = self._apify_client.dataset(actor_call["defaultDatasetId"]).list_items(clean=True).items

        return self._process_search_results(search_results)
    
    async def aget_search_results(self, search_query, country_code, lang ):
    
        if not search_query:
            raise ValueError("No value specified for search query")
        
        logging.info("Running Web search ...")
        search_input = await self.prepare_search_input(search_query, country_code,lang)
        actor = self._apify_client_async.actor(self._actor_id)
        actor_call =await actor.call(run_input = search_input)
        search_results = self._apify_client.dataset(actor_call["defaultDatasetId"]).list_items(clean=True).items

        return await self._aprocess_search_results(search_results)
    
    def _process_search_results(self,search_results):
        cleaned_results = dict()
        required_fields =('title', 'description', 'url')

        if search_results[0]['organicResults']:
            responses = search_results[0]['organicResults']
            for response in responses:
                for item in required_fields:
                    if item not in cleaned_results:
                        cleaned_results[item] = [response.get(item, "")]
                    else:
                        cleaned_results[item].append(response.get(item, ""))
                    
            return cleaned_results
        
        logging.info("No Search results found for the input query")
        return cleaned_results 

    @staticmethod
    async def _aprocess_search_results(search_results):
        cleaned_results = dict()
        required_fields =('title', 'description', 'url')

        if search_results[0]['organicResults']:
            responses = search_results[0]['organicResults']
            for response in responses:
                for item in required_fields:
                    if item not in cleaned_results:
                        cleaned_results[item] = [response.get(item, "")]
                    else:
                        cleaned_results[item].append(response.get(item, ""))
                    
            return cleaned_results
        
        logging.info("No Search results found for the input query")
        return cleaned_results    

    @staticmethod
    async def prepare_search_input(search_query, country_code ='us', lang ='en' ):
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


if __name__ == '__main__':
    import asyncio
    import time
    actor_id = os.getenv('APIFY_ACTOR_ID')
    api_token = os.getenv('APIFY_API_KEY')

    from config.core import config
    from search.search_utils import langs_2_search, prepare_seach_queries
    from translation.query_translation import abatch_query_translation
    
    async def main():
        #Fetch language to conduct search based on country domain, and choosen languages
        search_langs = langs_2_search(config.country, config.languages)

        #Prepare the search queries based on the inputs
        search_queries = prepare_seach_queries(config.name, 
                                            config.custom_search_query, 
                                            config.location, 
                                            config.add_keywords)
        print(search_queries)
        
        # Translated queries
        translated_queries = await abatch_query_translation(search_langs, 
                                                    search_queries)
        # Conduct web search

        AGS = ApifyGoogleSearch()
        search_results =await AGS.aperform_batch_searches(translated_queries, config.country)

        return search_results
    # country_code ='United States'
    # # lang='en'
    # search_queries = ["Who is vijay mallya?"]
    # t_queries =batch_query_translation(['hi', 'en'], search_queries)
    # # t_queries = [('en', 'news on "vijay mallya" related to money laundering, fraud or corruption charges'), ('en', 'criminal cases on "vijay mallya" in india'), ('en', '"vijay mallya" involvment in illegal activities : fraud'), ('en', 'adverse media about "vijay mallya" for alleged crimes in india'), ('en', "vijay mallya's crimes, scandals and bank frauds.")]
    # AGS = ApifyGoogleSearch(actor_id, api_token)
    # print(t_queries)

    # async def main():
    #     sres = await AGS.aperform_batch_searches(t_queries, country_code)
    #     return sres

    start =time.time()        
    sr =asyncio.run(main())
    print(sr)
    end =time.time()
    print(f"Time taken {(end-start)/60}")
    # print(len(sr['title']))