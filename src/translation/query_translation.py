import logging
import itertools
import asyncio

from google.cloud import translate_v2

def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language"""
    translate_client = translate_v2.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    result = translate_client.translate(text, target_language=target, format_='text')

    return result["translatedText"]

async def atranslate_text(target: str, text: str) -> dict:
    """Translates text into the target language"""
    translate_client = translate_v2.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    result = translate_client.translate(text, target_language=target, format_='text')

    return result["translatedText"]


def batch_query_translation(target_languages, search_queries):
    ''' Translate batch of queries to the different target languages'''

    if target_languages is None or search_queries is None:
        raise Exception(f"Either target languages or search queries missing. ")
    
    translated_queries =[]
    for query in search_queries:
        for lang in target_languages:
            translated_queries.append((lang, translate_text(lang, query.lower())))
    logging.info(f"Translated Queries: {translated_queries}")
    return translated_queries

async def abatch_query_translation(target_languages, search_queries):
    ''' Translate batch of queries to the different target languages'''

    if target_languages is None or search_queries is None:
        raise Exception(f"Either target languages or search queries missing. ")

    t_langs, coros =zip(*[(l_q_pair[0],atranslate_text(l_q_pair[0], l_q_pair[1].lower()))
            for l_q_pair in list(itertools.product(target_languages, search_queries))])

    t_results = await asyncio.gather(*coros)
    translated_queries = [*zip(t_langs, t_results)]

    logging.info(f"Translated Queries: {translated_queries}")
    return translated_queries


if __name__ == '__main__':
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parents[2] / '.env'
    load_dotenv(env_path)
    import time
    search_langs = ['th', 'km', 'en']
    search_query = ['News on "Main Babu" related to Money laundering, fraud or corruption charges', 'Criminal cases on "Main Babu" ', '"Main Babu" involvment in illegal activities ', 'Adverse media about "Main Babu" for alleged crimes ', 'Articles about Mani Babu involved in Money mule activities']
    async def main():
        res = await abatch_query_translation(search_langs, search_query)
        return res
    
    start =time.time()
    res =batch_query_translation(search_langs, search_query)

    # res =asyncio.run(main())
    
    end =time.time()
    print(res)
    print(f"time taken: {(end-start)/60}")

