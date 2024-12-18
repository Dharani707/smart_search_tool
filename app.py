import streamlit as st
import requests
import os
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext, load_index_from_storage


def scraping():
    page_no = 1
    data = []

    while page_no <= 8:

      url = "https://courses.analyticsvidhya.com/collections?page="+str(page_no)
      page = requests.get(url)
      soup = BeautifulSoup(page.text, 'html.parser')
      books = soup.find_all('li', class_ = 'products__list-item')
      book_no = 1

      for book in books:
        item = {}
        course_fees = book.find('span', class_ = 'course-card__price').text

        if book.find('h3').text.find("Launching Soon!") != -1:
          continue
        if course_fees == 'Free':
          soup_desc = BeautifulSoup(requests.get("https://courses.analyticsvidhya.com"+book.find('a').attrs['href']).text, 'html.parser')
          soup_sub_desc = BeautifulSoup(requests.get("https://courses.analyticsvidhya.com"+book.find('a').attrs['href']).text, 'html.parser')
          desc = soup_desc.find('div', class_ = "fr-view")
          sub_desc = soup_desc.find('header', class_ = "section__headings")

          item['BOOK_TITLE'] = book.find('h3').text
          item['COURSE_LINK'] = "https://courses.analyticsvidhya.com"+book.find('a').attrs['href']
          item['DESCRIPTION'] = sub_desc.find('h2').text+"\n"+desc.text

          data.append(item)

          book_no += 1

      page_no += 1
    
    documents = [Document(text = d['DESCRIPTION'], metadata={'title': d['BOOK_TITLE'], 'link': d['COURSE_LINK']}) for d in data]
    return documents

Settings.llm = None
Settings.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

def vector_index(documents):
    Settings.chunk_size = 1024
    index = VectorStoreIndex.from_documents(documents, 
                                            embed_model=Settings.embed_model, 
                                            chunk_size=Settings.chunk_size, 
                                            show_progress=True)
    return index

PERSIST_DIR = "vector_storage/storage"

def session_state():
    
    if 'vector_index' not in st.session_state:
        if os.path.exists(PERSIST_DIR):
            st.session_state.storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            st.session_state.vector_index = load_index_from_storage(st.session_state.storage_context)
        else:
            st.write("there is no path.")
    return st.session_state.vector_index
   

def handle_query(index,query):
    vector_retriever = VectorIndexRetriever(index = index, similarity_top_k = 6)
    query_engine = RetrieverQueryEngine(retriever = vector_retriever)
    response = query_engine.query(query)  
    if response:
        for x in response.source_nodes:
            st.write("Course:",x.node.metadata['title'])
            st.write("Link:",x.node.metadata['link'])
            st.write("\n\n\n\n")


input_text = st.text_input("Enter your requirements of skills to be developed or the course which you are interested")
st.caption("eg: can you list out courses related to Generative AI")
st.title("Analytics Vidhya's Smart search system")
st.write("This app will gives the informations of free courses related to AI/ML, Data Science, and Analytics from Analytics Vidhya")
if st.button('Scrape and Search'):
    index = session_state()
    if input_text:
        st.write("Here are the top 6 free courses from Analytics Vidhya. To explore more free and paid courses, please visit the Analytics Vidhya website")
        handle_query(index, input_text)
        st.write("For more details visit Analytics Vidhya website")
    else:
        st.write("Please enter a query in the text box.")
  
  

