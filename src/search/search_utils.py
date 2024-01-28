import logging
from utils.utils_data import load_lang_dict, load_country_dict, load_ctry_lang_mapping

def default_search_queries(name, location, other_keywords):
    ''' Define default search queries'''

    q1 = f'News on "{name}" related to Money laundering, fraud or corruption charges'
    q2 = f'Criminal cases on "{name}" { "in "+location if location is not None else ""}'
    q3 = f'"{name}" involvment in illegal activities { ": "+other_keywords if other_keywords is not None else ""}'
    q4 = f'Adverse media about "{name}" for alleged crimes { "in "+location if location is not None else ""}'
    return [q1, q2, q3,q4]

def prepare_seach_queries(name, custom_query, location, other_keywords ):
    '''Prepare search queries to perform web search'''

    if (not name) and (not custom_query):
        raise ValueError("Neither 'Name' was provided not 'Custom search' Query was provided")
    
    if name and custom_query:
        default_queries =default_search_queries(name,location,other_keywords)
        default_queries.append(custom_query)
        logging.info(f'Search queries: {default_queries}')
        return default_queries
    elif name:
        default_queries =default_search_queries(name,location,other_keywords)
        logging.info(f'Search queries: {default_queries}')
        return default_queries
    else:
        logging.info(f'Search queries: {[custom_query]}')
        return [custom_query]
    
def langs_2_search(domain_ctry ='United States', choosen_langs:list = None, static_lang ='English'):
    '''Prepare list of languages the search should be performed'''

    lang_dict =load_lang_dict()
    ctry_dict =load_country_dict()
    lang_ctry_dict = load_ctry_lang_mapping()

    ctry_code = ctry_dict[domain_ctry]
    ctry_langs = lang_ctry_dict[ctry_code] # get official languages of country
    
    if choosen_langs is None:
        logging.info(f"No specific langauge choosen, hence defaulting to official language of {domain_ctry}")
        return [lang_dict[static_lang], *ctry_langs] 

    if not isinstance(choosen_langs, list):
        raise TypeError(f'{type(choosen_langs)} passed instead of type list for language attribute')
    
    langs_to_search =[lang_dict[static_lang],*ctry_langs]
    for lang in choosen_langs:
        langs_to_search.append(lang_dict[lang])
    return list(set(langs_to_search))

if __name__ == '__main__':
    name ='Main Babu'
    location = 'India'
    other_keywords = 'Fraud'
    query ='Articles about Mani Babu involved in Money mule activities'
    print(prepare_seach_queries(name, query, None, None))
    print(langs_2_search('Thailand', ['Cambodian']))