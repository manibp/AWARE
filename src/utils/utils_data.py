import time
from collections import defaultdict
import re

from config.core import config
from config import core
import json


def load_lang_dict():
    '''Load ISO 639-1 language code dictionary'''

    lang_dict_path = core.DATASET_DIR / config.language_dict
    with open(lang_dict_path, 'r') as f:
        lang_dict =json.loads(f.read())

    return lang_dict

def load_country_dict():
    '''Load country code dictionary'''

    ctry_dict_path = core.DATASET_DIR / config.country_dict
    with open(ctry_dict_path, 'r') as f:
        ctry_dict =json.loads(f.read())

    return ctry_dict

def load_ctry_lang_mapping():
    '''Load country to language mapping'''

    ctry_lang_mapping_path = core.DATASET_DIR / config.ctry_lang_map
    with open(ctry_lang_mapping_path, 'r') as f:
        ctry_lang_mapping =json.loads(f.read())

    return ctry_lang_mapping

def time_deco(id): 
    def outer_wrapper(func):      
        def wrapper(*args, **kwargs):
            nonlocal id
            start = time.time()
            result = func(*args, **kwargs)
            time_elapsed =  time.time() - start
            print(f"Time elapsed for {func.__name__} is {time_elapsed}")
            # with open('./func_time_log.txt', 'a+') as f:
            #     f.write(f" Run Id: {id}. Time elapsed for {func.__name__} is {time_elapsed} \n")
            return result
        
        return wrapper
    
    return outer_wrapper

def merge_dicts(*dicts):
    
    result_dict =defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            result_dict[key].extend(value)
    
    return dict(result_dict)

def pp_retriever_query_results(results):
    ''' Clean up the responses obtained from LLM while rewriting the retriever queries'''
    pattern = re.compile(r'[([{].*?[)}\]]', re.DOTALL)
    for i in range(len(results)):
       results[i] = re.sub(pattern, '', results[i]).replace('\n', '')

    return results


if __name__ == '__main__':
    country_dict =load_country_dict()
    print(country_dict['India'])
