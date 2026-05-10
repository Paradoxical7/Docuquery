# DocuQuery — AI-Powered Document Q&A

Upload any PDF and ask questions about it in plain English. DocuQuery uses RAG (Retrieval-Augmented Generation) to find relevant passages and generate accurate answers using OpenAI.

## Features
- Upload any PDF document
- Ask natural language questions about its content
- Real-time answers powered by GPT-3.5
- Clean chat interface

## Tech Stack
- Python, Flask
- OpenAI API (GPT-3.5)
- ChromaDB (vector storage)
- PyMuPDF (PDF parsing)

## How it works
1. PDF is parsed and split into chunks
2. Chunks are embedded and stored in ChromaDB
3. User question retrieves the most relevant chunks
4. Retrieved context + question is sent to OpenAI
5. Answer is returned in the chat interface

## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies