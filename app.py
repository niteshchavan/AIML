from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from flask import Flask, request, render_template, jsonify

import re
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
chroma_db = 'chromadb'

url = "https://www.pitambari.com/"
loader = RecursiveUrlLoader(
    url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()
print(len(docs))

#print(docs)
#print(docs[0].page_content)

def clean_text(text):
    return re.sub(r'\n+', ' ', text).strip()

#cleaned_docs = clean_text(docs.page_content)
#print(cleaned_docs)

results = clean_text([doc.page_content for doc in docs])
