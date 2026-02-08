from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import pprint

filepath="./example.pdf"

loader=PyPDFLoader(filepath)

docs=loader.load()

""" we use the RecursiveCharacterTextSplitter to maintain the context and paragraphs intact"""

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

chunks=text_splitter.split_documents(docs)

for idx,chunk in enumerate(chunks):
  chunk.metadata["chunk_id"]=idx


embedings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_db=FAISS.from_documents(
  documents=chunks,
  embedding=embedings
)

faiss_db.save_local("ragdata")

query="What are the key steps involved in the information security risk assessment process?"
topk=3

def queryresponse(query,topk):
  docs=faiss_db.similarity_search_with_score(query,k=topk)

  print(f"\n\nQuery :{query}")
  print(f" Top K :{topk}")
  print("\n\n")

  for index,(d,score) in enumerate(docs):
    print("-"*40)
    print(f"Chunk :{index+1}\n")
    print(f"Chunk Content:\n{d.page_content}")
    print(f"\n Source:{d.metadata['source']}")
    print(f"\n Page : {d.metadata['page']}")
    print(f"\n chunk_id : {d.metadata['chunk_id']}")
    print(f"\n Score : {score}")


print("Sample output:")
queryresponse(query,topk)

print("The above output is the sample output Please enter your query below...")

query=input("Enter the Query: ")
topk=int(input("Enter the value of TopK :"))
queryresponse(query,topk)