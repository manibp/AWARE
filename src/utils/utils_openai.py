
import re

def pp_retriever_query_results(results):
    ''' Clean up the responses obtained from LLM while rewriting the retriever queries'''
    pattern = re.compile(r'[([{].*?[)}\]]', re.DOTALL)
    for i in range(len(results)):
       results[i] = re.sub(pattern, '', results[i]).replace('\n', '')

    return results
