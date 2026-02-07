from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import pprint

filepath="./example.pdf"

loader=PyPDFLoader(filepath)
print(loader)

docs=loader.load()


""" we use the RecursiveCharacterTextSplitter to maintain the context and paragraphs intact"""

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)



chunks=text_splitter.split_documents(docs)

for idx,chunk in enumerate(chunks):
  chunk.metadata["chunk_id"]=idx


embedings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts=[chunk.page_content for chunk in chunks]
metadata=[chunk.metadata for chunk in chunks]
ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks]

vectors=embedings.embed_documents(texts)

faiss_db=FAISS.from_documents(
  documents=chunks,
  embedding=embedings
)

faiss_db.save_local("ragdata")

query="How does top management demonstrate leadership and commitment to the ISMS?"
docs=faiss_db.similarity_search_with_score(query,k=5)

for index,(d,score) in enumerate(docs):
  print("-"*40)
  print(f"Chunk :{index+1}\n")
  print(f"Chunk Content:\n{d.page_content}")
  print(f"\n Source:{d.metadata['source']}")
  print(f"\n Page : {d.metadata['page']}")
  print(f"\n chunk_id : {d.metadata['chunk_id']}")
  print(f"\n Score : {score}")