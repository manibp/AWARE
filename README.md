
# AWARE: Adverse Media Assessment and Reporting Engine

AWARE is a powerful streamlit application designed for efficient reporting of adverse media associated with an individual. The tool seamlessly integrates various services to provide accurate and relevant articles related to the provided query.

## Key Features

- **Multilingual Search:** Translates the input query to different languages based on the web search country domain.
- **Apify Integration:** Performs a multilingual search using the Apify API to retrieve results from various sources.
- **Pinecone Integration:** Utilizes a large language model (LLM) for vector embeddings stored in Pinecone Vector database.
- **Similarity Scoring:** Calculates similarity scores between the input query and search results using semantic search.
- **Streamlit Interface:** Interactive user interface hosted on Streamlit public cloud.

## Application URL

Visit the AWARE application hosted on Streamlit public cloud: [AWARE App](https://adversemediacheck.streamlit.app/#aware-adverse-media-assessment-and-reporting-engine)

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
3. Run Streamlit Application
   streamlit run src/app.py


Snapshot of Streamlit App. 
<img width="1220" alt="image" src="https://github.com/manibp/AWARE/assets/14993216/3049db40-5dad-4b61-bd5d-9416966b2e42">


