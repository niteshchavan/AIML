from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


#a list of urls
urls = ["https://langchain-ai.github.io/langgraph/"]

#initialize document loader
loader = AsyncChromiumLoader(urls, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
docs = loader.load()
print(docs)

bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"])
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)

splits = splitter.split_documents(docs_transformed)

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

print(splits)
#extracted_content = extract(schema=schema, content=splits[0].page_content)

