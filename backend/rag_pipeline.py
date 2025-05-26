from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from typing import Optional
import logging
import json
from datetime import datetime
import shutil
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (assuming this file is in backend/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Config
JSON_FILE_PATH = os.getenv(
    "KNOWLEDGE_FILE_PATH", str(PROJECT_ROOT / "city_information" / "knowledge.json")
)
VECTORSTORE_PATH = os.getenv(
    "VECTORSTORE_PATH", str(PROJECT_ROOT / "data" / "vectorstore")
)
BACKUP_PATH = os.getenv(
    "BACKUP_PATH", str(PROJECT_ROOT / "data" / "vectorstore_backups")
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MODEL_NAME = "llama3.2"


def validate_knowledge_file():
    """Validate the structure of the knowledge file."""
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Knowledge file must contain a dictionary")
            if "knowledge_base" not in data:
                raise ValueError("Knowledge file must contain a 'knowledge_base' key")
            if not isinstance(data["knowledge_base"], dict):
                raise ValueError("'knowledge_base' must be a dictionary")
            if not data["knowledge_base"]:
                raise ValueError("'knowledge_base' is empty")
            return True
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in knowledge file")
    except Exception as e:
        raise ValueError(f"Error validating knowledge file: {str(e)}")


def backup_vectorstore():
    """Create a backup of the vectorstore."""
    if os.path.exists(VECTORSTORE_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(BACKUP_PATH, f"backup_{timestamp}")
        shutil.copytree(VECTORSTORE_PATH, backup_dir)
        logger.info(f"Created vectorstore backup at {backup_dir}")


def cleanup_vectorstore():
    """Clean up old vectorstore backups."""
    try:
        if os.path.exists(BACKUP_PATH):
            # Keep only the 5 most recent backups
            backups = sorted(
                [d for d in os.listdir(BACKUP_PATH) if d.startswith("backup_")]
            )
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    backup_path = os.path.join(BACKUP_PATH, old_backup)
                    shutil.rmtree(backup_path)
                    logger.info(f"Removed old backup: {old_backup}")
    except Exception as e:
        logger.error(f"Error cleaning up vectorstore backups: {str(e)}")


def get_knowledge_last_modified():
    """Get the last modification time of the knowledge file."""
    try:
        return os.path.getmtime(JSON_FILE_PATH)
    except:
        return 0


def should_rebuild_vectorstore():
    """Check if vectorstore needs to be rebuilt."""
    if not os.path.exists(VECTORSTORE_PATH):
        return True

    # Check if knowledge file is newer than vectorstore
    knowledge_mtime = get_knowledge_last_modified()
    vectorstore_mtime = os.path.getmtime(VECTORSTORE_PATH)

    return knowledge_mtime > vectorstore_mtime


def load_and_prepare_vectorstore():
    try:
        # Validate knowledge file
        validate_knowledge_file()

        # Create necessary directories
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        os.makedirs(BACKUP_PATH, exist_ok=True)

        # Check if vectorstore exists and is valid
        vectorstore_exists = os.path.exists(
            os.path.join(VECTORSTORE_PATH, "index.faiss")
        )

        # Check if we need to rebuild the vectorstore
        if vectorstore_exists and not should_rebuild_vectorstore():
            logger.info(f"Loading existing vectorstore from {VECTORSTORE_PATH}")
            try:
                embeddings = OllamaEmbeddings(model=MODEL_NAME)
                return FAISS.load_local(
                    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.warning(f"Failed to load existing vectorstore: {str(e)}")
                logger.info("Will create new vectorstore")
                vectorstore_exists = False

        # If vectorstore doesn't exist or needs rebuilding
        if not vectorstore_exists:
            # Backup existing vectorstore before rebuilding if it exists
            if os.path.exists(VECTORSTORE_PATH):
                backup_vectorstore()
                # Clean up old vectorstore
                shutil.rmtree(VECTORSTORE_PATH)
                os.makedirs(VECTORSTORE_PATH, exist_ok=True)

            logger.info(f"Building new vectorstore from {JSON_FILE_PATH}")
            loader = JSONLoader(
                file_path=JSON_FILE_PATH,
                jq_schema=".knowledge_base[] | .[]",
                text_content=False,
            )
            documents = loader.load()

            logger.info(f"Splitting {len(documents)} documents into chunks")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            split_docs = splitter.split_documents(documents)

            logger.info("Initializing embeddings")
            embeddings = OllamaEmbeddings(model=MODEL_NAME)

            logger.info("Creating vector store")
            vectorstore = FAISS.from_documents(
                documents=split_docs, embedding=embeddings
            )

            # Save the vectorstore
            logger.info(f"Saving vectorstore to {VECTORSTORE_PATH}")
            vectorstore.save_local(VECTORSTORE_PATH)

            return vectorstore
    except Exception as e:
        logger.error(f"Error in load_and_prepare_vectorstore: {str(e)}")
        raise


def setup_rag_chain(vectorstore):
    try:
        logger.info("Setting up RAG chain")
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful city information assistant. Your task is to provide accurate and concise information about the city based on the given context.

Context: {context}

Question: {input}

Instructions:
1. Use only the information provided in the context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question"
3. Keep your answers concise and to the point
4. Include specific details when available
5. If you're not completely sure about something, indicate your uncertainty

Answer:"""
        )

        llm = ChatOllama(model=MODEL_NAME, temperature=0.1, timeout=60)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        logger.info("RAG chain setup complete")
        return retrieval_chain
    except Exception as e:
        logger.error(f"Error in setup_rag_chain: {str(e)}")
        raise
