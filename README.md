
# AWARE: Adverse Media Assessment and Reporting Engine

AWARE is a powerful streamlit application designed for efficient reporting of adverse media associated with an individual. The tool seamlessly integrates various services to provide accurate and relevant articles related to the provided query.

## Key Features

- **Multilingual Search:** Translates the input query to different languages based on the web search country domain.
- **Apify Integration:** Performs asynchronous multilingual search using the Apify API to retrieve results from various sources.
- **Pinecone Integration:** Utilizes a large language model (LLM) for vector embeddings stored in Pinecone Vector database.
- **Query Rewriting:** Refines search accuracy by rephrasing search queries with custom prompts in an interrogative manner using GPT-4, ensuring precise retrieval of relevant information.
- **Similarity Scoring:** Computes similarity scores between the input query and search results using hybrid search that combines both lexical and vector search.
- **Streamlit Interface:** Interactive frontend applications to perform search.
- **Chat Assistant:** Engages in natural language conversations with your data using the AI Assistant built on LlamaIndex ChatEngine

## Repository Structure

- `/src`: Main script for the Streamlit application (`app.py`).
- `/data`: Mapping files, including country-language mapping, language dictionary, country dictionary, etc.
- `requirements.txt`: List of dependencies for running the application.

## Usage

1. **Clone Repository:**
   ```bash
   git clone https://github.com/manibp/AWARE.git
   cd AWARE
2. Install Dependencies
   pip install -r requirements.txt
3. Set API keys of following services as Environment variables
   - OpenAI, Huggingface, Pinecone, Apify, Google Translation Credentials
5. Navigate to src folder and run Streamlit Application
   python3 -m streamlit run app.py


Snapshot of Streamlit App. 
<img width="1512" alt="image" src="https://github.com/manibp/AWARE-v1/assets/14993216/8192857d-ec94-405e-9135-460a48950ced">

<img width="1512" alt="image" src="https://github.com/manibp/AWARE-v1/assets/14993216/d1e0ec4a-b6c5-4d17-aee0-c1eee4fd7b7d">


