from langchain_community.document_loaders import PyPDFLoader
import os
import glob

# Method 1: Using glob to find all PDF files
pdf_files = glob.glob("./paper/*.pdf")

docs = []

for pdf_file in pdf_files:
    print(f"Loading: {pdf_file}")
    loader = PyPDFLoader(
        file_path=pdf_file,
        mode="single",
        pages_delimiter=""
    )
    
    docs_lazy = loader.lazy_load()
    
    for doc in docs_lazy:
        docs.append(doc)

print(f"Total documents loaded: {len(docs)}")

if docs:
    print(f"\nFirst document content preview:")
    print(docs[0].page_content[:100])
    print(f"\nFirst document metadata:")
    print(docs[0].metadata)