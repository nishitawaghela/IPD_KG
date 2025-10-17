# evaluate_extraction.py

import json
from datasets import load_dataset
from tqdm import tqdm

# We import the specific function we want to test from our main processing script
from processing import extract_knowledge_with_llm

def format_relations_for_comparison(entities, relations):
    """
    Converts SciERC's relation format into a standardized set of tuples for easy comparison.
    Example output: {('method', 'used-for', 'task'), ('material', 'hyponym-of', 'generic')}
    """
    formatted_relations = set()
    entity_map = {i: " ".join(entities[i]).lower() for i in range(len(entities))}
    
    for rel_data in relations:
        source_name = entity_map.get(rel_data[0])
        target_name = entity_map.get(rel_data[1])
        relation_type = rel_data[2].lower()
        
        if source_name and target_name:
            formatted_relations.add((source_name, relation_type, target_name))
            
    return formatted_relations

def evaluate_knowledge_extraction():
    """
    Runs the evaluation on the SciERC dataset and prints the performance metrics
    for the knowledge extraction function.
    """
    print("Loading SciERC dataset from Hugging Face...")
    try:
        dataset = load_dataset("scierc", "scierc", split="test")
    except Exception as e:
        print(f"Failed to load dataset. Check your internet connection. Error: {e}")
        return

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # For initial testing, you might want to process only a small subset
    # To run on the full dataset, remove the [:20] slice
    subset_dataset = dataset #.select(range(20)) 
    
    print(f"Starting evaluation on {len(subset_dataset)} documents...")
    
    for item in tqdm(subset_dataset, desc="Evaluating Documents"):
        full_text = " ".join(item['tokens'])
        
        # 1. Get the TRUE relations from the dataset's annotations
        true_relations = format_relations_for_comparison(item['entities'], item['relations'])
        
        # 2. Get the PREDICTED relations from our LLM pipeline
        predicted_kg_data = extract_knowledge_with_llm(full_text)
        
        if predicted_kg_data and 'relationships' in predicted_kg_data:
            predicted_relations_list = [
                (rel.get('source','').lower(), rel.get('type','').lower(), rel.get('target','').lower())
                for rel in predicted_kg_data['relationships']
            ]
            predicted_relations = set(predicted_relations_list)
        else:
            predicted_relations = set()
            
        # 3. Compare the sets to find TPs, FPs, and FNs for this document
        true_positives = len(predicted_relations.intersection(true_relations))
        false_positives = len(predicted_relations.difference(true_relations))
        false_negatives = len(true_relations.difference(predicted_relations))
        
        # 4. Accumulate the total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    # --- Calculate Final Metrics ---
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Knowledge Extraction Benchmark Results ---")
    print(f"Total Relations in Dataset: {total_true_positives + total_false_negatives}")
    print(f"Total Relations Extracted by LLM: {total_true_positives + total_false_positives}")
    print("---------------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("---------------------------------------------")
    print("Note: Precision = Of the relations extracted, how many were correct.")
    print("      Recall    = Of all correct relations, how many were found.")

if __name__ == "__main__":
    evaluate_knowledge_extraction()