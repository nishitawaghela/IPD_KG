import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from neo4j import GraphDatabase
import json
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
# Load environment variables from .env file
load_dotenv()

# Configure API keys and models
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. FILE AND TEXT PROCESSING FUNCTIONS ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file."""
    print(f"Reading text from {pdf_path}...")
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    print("Text extraction complete.")
    return text

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> list[str]:
    """Splits text into smaller chunks."""
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

# --- 3. KNOWLEDGE GRAPH FUNCTIONS (UPDATED FOR GROQ) ---

def extract_knowledge_with_llm(text_chunk: str, model_name="llama-3.1-8b-instant"):
    """Uses the Groq API to extract entities and relationships from a text chunk."""

    # --- DEFINE IT HERE ---
    system_prompt = """
    You are an expert in knowledge graph construction. From the text below, extract key entities and the relationships between them.
    Format the output as a single, raw JSON object with two keys: "nodes" and "relationships".
    - "nodes" should be a list of objects, each with a "name" and a "type".
    - "relationships" should be a list of objects, each with a "source", "target", and "type".
    Do not include any text or explanations outside of the JSON object.
    """

    user_prompt = f"Text to analyze:\n---\n{text_chunk}\n---"

    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt}, # <-- USE IT HERE (check spelling!)
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        json_response_text = response.choices[0].message.content
        return json.loads(json_response_text)
    except Exception as e:
        print(f"Error extracting knowledge from chunk with Groq: {e}")
        return None

class Neo4jGraph:
    """Class to interact with the Neo4j database."""
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def populate_graph(self, kg_data: dict):
        """Populates the graph with nodes and relationships."""
        if not kg_data or 'nodes' not in kg_data or 'relationships' not in kg_data:
            return
        
        with self._driver.session() as session:
            for node in kg_data['nodes']:
                session.run("MERGE (n:Concept {name: $name, type: $type})", name=node['name'], type=node.get('type', 'Unknown'))
            
            for rel in kg_data['relationships']:
                session.run("""
                    MATCH (a:Concept {name: $source})
                    MATCH (b:Concept {name: $target})
                    MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                """, source=rel['source'], target=rel['target'], type=rel.get('type', 'related'))

# --- 4. ORCHESTRATOR FUNCTION ---

def process_document(pdf_path: str):
    """The main orchestrator function for the processing pipeline."""
    full_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(full_text)
    
    neo4j_graph = Neo4jGraph(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    total_embeddings = []
    
    for i, chunk in enumerate(text_chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(text_chunks)} ---")
        
        print("Generating embedding...")
        embedding = embedding_model.encode(chunk)
        total_embeddings.append(embedding)
        
        print("Extracting knowledge with Groq...")
        kg_data = extract_knowledge_with_llm(chunk)
        if kg_data:
            print(f"Found {len(kg_data['nodes'])} nodes and {len(kg_data['relationships'])} relationships.")
            neo4j_graph.populate_graph(kg_data)
        else:
            print("No knowledge extracted from this chunk.")
            
    neo4j_graph.close()
    print("\n--- Document Processing Complete ---")
    return {"chunks_processed": len(text_chunks), "embeddings_generated": len(total_embeddings)}