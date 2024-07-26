import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


loader = RecursiveUrlLoader("https://math.berkeley.edu/wp/", extractor=bs4_extractor)
docs = loader.load_and_split()
print([doc.page_content for doc in docs])




# Split the cleaned documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
pages = loader.load_and_split()
chunks = text_splitter.split_documents(pages)

#print(docs)