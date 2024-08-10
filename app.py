import gradio as gr
import os
import time
import arxiv
import ollama
import chromadb
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_papers(query, question_text):
    dirpath = "arxiv_papers"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    print("Starting arXiv search...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=20,
        sort_order=arxiv.SortOrder.Descending
    )

    papers_metadata = []
    for result in client.results(search):
        metadata = {
            'title': result.title,
            'authors': ', '.join([author.name for author in result.authors]),
            'url': result.pdf_url,
            'date': result.published.date().strftime('%Y-%m-%d')  # Convert date to string
        }
        try:
            print(f"Downloading paper: {result.title}")
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            papers_metadata.append(metadata)
        except (FileNotFoundError, ConnectionResetError) as e:
            print("Error occurred:", e)
            time.sleep(5)

    print("Loading papers...")
    loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
    papers = []
    try:
        papers = loader.load()
        print(f"{len(papers)} papers loaded.")
    except Exception as e:
        print(f"Error loading file: {e}")

    full_text = ''
    for paper in papers:
        full_text += paper.page_content

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])
    print(f"Text split into {len(paper_chunks)} chunks.")

    print("Initializing ChromaDB...")
    client = chromadb.Client()
    collection = client.create_collection(name="arxiv_papers")

    print("Generating embeddings and storing in ChromaDB...")
    for i, chunk in enumerate(paper_chunks):
        response = ollama.embeddings(
            model='nomic-embed-text:latest',
            prompt=chunk.page_content
        )
        embedding = response["embedding"]

        # Ensure metadata is properly linked with text chunks
        chunk_metadata = papers_metadata[min(i, len(papers_metadata) - 1)]

        # Ensure metadata dictionary is not empty or missing required fields
        if not all(key in chunk_metadata for key in ['title', 'authors', 'url', 'date']):
            chunk_metadata = {
                'title': 'Unknown Title',
                'authors': 'Unknown Authors',
                'url': 'Unknown URL',
                'date': 'Unknown Date'
            }

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[chunk_metadata]
        )
        print(f"Stored chunk {i+1} of {len(paper_chunks)}")

    print("Generating embedding for the query and retrieving the most relevant document...")
    response = ollama.embeddings(
        prompt=question_text,
        model='nomic-embed-text:latest'
    )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )
    
    if results and results['documents']:
        context_data = results['documents'][0][0]
        context_metadata_list = results['metadatas'][0]  # Get the first metadata entry from the list
        print("Relevant document found.")
    else:
        context_data = "No relevant documents found."
        context_metadata_list = [{
            'title': 'Unknown Title',
            'authors': 'Unknown Authors',
            'url': 'Unknown URL',
            'date': 'Unknown Date'
        }]
        print("No relevant documents found.")

    # Generate references from metadata
    references = []
    for context_metadata in context_metadata_list:
        reference = f"{context_metadata.get('title', 'Unknown Title')} by {context_metadata.get('authors', 'Unknown Authors')} ({context_metadata.get('date', 'Unknown Date')}) - [PDF]({context_metadata.get('url', 'Unknown URL')})"
        references.append(reference)

    references_text = "\n".join(references)

    # Use the LLaMA model (llama3.1) for generating the final response
    prompt_text = f"Using this data: {context_data}. Respond to this prompt: {question_text}"
    final_response = ollama.generate(
        model='llama3.1',  # Specify the LLaMA model here
        prompt=prompt_text
    )['response']

    final_output = f"{final_response}\n\n**References:**\n{references_text}"

    print("Final response generated.")
    return final_output

# Set up the Gradio interface
iface = gr.Interface(
    fn=process_papers,
    inputs=["text", "text"],
    outputs="text",
    description="Enter a search query and a question to process arXiv papers, with sources referenced."
)

iface.launch()
