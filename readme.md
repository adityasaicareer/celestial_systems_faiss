# Documentation For FAISS 

**This code will run in both Python3.11 and Python3.14**

**Libraries Required**

```
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import pprint
```

#### The Above Libraries were used in the code

* **langchain_community.document_loaders import PyPDFLoader** helps us toparse the PDF documents required for the context

* **langchain_text_splitters import RecursiveCharacterTextSplitter**. This will Help us to create Chunks from the parsed PDF document data

* **langchain_huggingface import HuggingFaceEmbeddings** helps us to use the **sentence-transformers/all-MiniLM-L6-v2** an Industry Standard to create the embedings

* **langchain_community.vectorstores import FAISS** helps us to get the FAISS(FAcebook AI Similarity Search) used to store the Embedings and perform Fast Similarity Search