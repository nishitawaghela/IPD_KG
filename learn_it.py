import os
from neo4j import GraphDatabase
from groq import Groq

# Initialize connections based on your Tech Stack [cite: 322, 323]
# Ensure these match your processing.py credentials
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_concept_graph_context(concept_name):
    """
    Retrieves the specific subgraph for a concept to ensure factual grounding.
    Implements the logic from Section 3.3 (Knowledge Graph Representation)[cite: 387].
    """
    # Cypher query to find the concept and its immediate relationships (1-hop)
    query = """
    MATCH (c:Concept {id: $name})-[r]-(neighbor)
    RETURN type(r) as relation, neighbor.id as connected_concept
    """
    
    with driver.session() as session:
        result = session.run(query, name=concept_name)
        relationships = [record.data() for record in result]
    
    if not relationships:
        return None

    # Format the graph data into a natural language context string
    # Transforming triples (Subject-Predicate-Object) into text [cite: 386]
    context_str = f"Concept: {concept_name}\nContext from Study Material:\n"
    for item in relationships:
        # e.g., "Soil HAS_COMPONENT Minerals"
        context_str += f"- {concept_name} {item['relation']} {item['connected_concept']}\n"
        
    return context_str

def generate_remediation_content(concept_name):
    """
    Uses the LLM to generate a personalized explanation based ONLY on the graph data.
    This aligns with Section 2.7 (Text Simplification)[cite: 287].
    """
    # 1. Retrieve the strict structure from Neo4j
    graph_context = get_concept_graph_context(concept_name)
    
    if not graph_context:
        return {
            "status": "error",
            "message": f"Concept '{concept_name}' not found in the Knowledge Graph."
        }

    # 2. Prompt the LLM (Llama-3 via Groq) acting as the 'Smart Tutor' [cite: 246]
    system_prompt = (
        "You are an AI Tutor in an Adaptive Learning System. "
        "Your goal is to explain a concept to a student who just failed a test on it. "
        "CRITICAL: You must ONLY use the provided Knowledge Graph context. "
        "Do not use outside knowledge. If the graph says 'Soil HAS_TYPE Clay', teach that exact fact."
    )
    
    user_prompt = f"""
    The student is weak in the concept: "{concept_name}".
    
    Here is the exact Knowledge Graph data extracted from their textbook:
    {graph_context}
    
    Please provide:
    1. A simple 2-sentence definition linking the concept to its neighbors.
    2. A bulleted list of key facts derived strictly from the relationships above.
    3. A short "Remember This" tip to help them answer reasoning questions next time.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3  # Keep it factual [cite: 270]
    )

    return {
        "status": "success",
        "concept": concept_name,
        "graph_data": graph_context, # Useful for debugging or showing the user the graph
        "lesson_content": response.choices[0].message.content
    }