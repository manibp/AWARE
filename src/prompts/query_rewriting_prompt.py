from langchain.prompts import PromptTemplate

query_rewrite_template = """ Rewrite the given query to extract comprehensive and detailed information on a specified subject. 
Maintain the original context and meaning while using interrogative forms to inquire about specific details related 
to the subject. Avoid introducing new subjects or altering the query's intent significantly and keep them concise.
Additionally, rank the rewritten queries from most relevant to least relevant on a scale of 1-5 (5 being the best and 1 worst)
based on how well they retain the context, intent, and directness of the original query and select only the best query. 
Ensure the rewritten queries adhere to a formal tone. Make sure the response generated should only contain the first best query
without extra information like rank number, new line characters and serial numbers.

query: {query_str}

response: """

query_rewrite_prompt =PromptTemplate.from_template(query_rewrite_template)
