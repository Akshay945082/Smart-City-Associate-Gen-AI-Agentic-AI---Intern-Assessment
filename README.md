# Smart-City-Associate-Gen-AI-Agentic-AI---Intern-Assessment
Smart City Information Assistant : Build an intelligent assistant for citizens to query information about city services, policies, and facilities using our provided Smart City Knowledge Base

# City Information Assistant

A RAG-based application that provides information about cities using LangChain, Ollama, and Streamlit.

## Project Structure

```
.
├── backend/                 # Backend FastAPI server
│   ├── main.py             # FastAPI application
│   ├── rag_pipeline.py     # RAG pipeline implementation
│   └── __init__.py         # Package initialization
├── fronted/                # Frontend Streamlit application
│   └── app.py             # Streamlit UI
├── city_information/       # Knowledge base
│   └── knowledge.json     # City information data
├── data/                  # Vector store and backups
│   ├── vectorstore/       # FAISS vector store
│   └── vectorstore_backups/ # Vector store backups
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running
3. Llama3.2 model pulled in Ollama

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On Unix/MacOS
   source myenv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull the required Ollama model:
   ```bash
   ollama pull llama3.2
   ```

## Running the Application

1. Start the backend server:
   ```bash
   uvicorn backend.main:app --reload
   ```

2. In a separate terminal, start the frontend:
   ```bash
   streamlit run fronted/app.py
   ```

3. Open your browser and navigate to:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

## API Endpoints

- `GET /health`: Check backend health
- `POST /query`: Ask questions about the city
- `GET /data`: Get raw city information data

## Features

- RAG-based question answering
- Real-time response with confidence scores
- Conversation history with delete functionality
- Automatic vector store management
- Backup system for vector store

## Troubleshooting

1. If you see "Method Not Allowed" error:
   - Ensure the backend is running
   - Check if you can access http://localhost:8000/health

2. If you see "Model not found" error:
   - Ensure Ollama is running
   - Verify llama3.2 is pulled: `ollama list`

3. If the frontend can't connect to the backend:
   - Check if both servers are running
   - Verify the API URLs in fronted/app.py 
