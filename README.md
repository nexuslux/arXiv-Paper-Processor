# arXiv Paper Processor

This project allows you to search for academic papers on arXiv, download and process them, and generate responses to specific questions using embeddings and language models. The application leverages several tools including Gradio for the interface, ChromaDB for embedding storage, and LangChain for text processing.

## Features

- **arXiv Paper Search**: Search for papers based on a query, download them in PDF format, and extract relevant metadata.
- **Text Processing**: Extract and split the content of downloaded papers into manageable chunks.
- **Embedding Generation**: Generate text embeddings using the `nomic-embed-text` model and store them in ChromaDB.
- **Document Retrieval**: Retrieve the most relevant document based on a query using the generated embeddings.
- **Response Generation**: Generate responses to specific questions using the LLaMA model, incorporating references to the source documents.

## Requirements

- Python 3.8+
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
