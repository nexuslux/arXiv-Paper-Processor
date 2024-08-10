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
  ```

Usage
Clone the Repository:

  ```bash
git clone https://github.com/yourusername/arXiv-Paper-Processor.git
  ```
cd arXiv-Paper-Processor
Install Dependencies:

  ```bash
pip install -r requirements.txt
  ```

## Run the Application:

  ```bash
python app.py
```
Access the Gradio Interface:
The interface will launch in your web browser. Enter your search query and question to start processing papers.

## Project Structure

app.py: Main application file containing the core logic.
requirements.txt: List of required Python packages.
README.md: Project documentation.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
