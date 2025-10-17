# evaluate_extraction.py

import json
from datasets import load_dataset
from tqdm import tqdm # A handy library for progress bars
import os # Make sure os is imported for getenv
from dotenv import load_dotenv # Make sure dotenv is imported

# Important: We import the function we want to test from our existing processing.py file
from processing import extract_knowledge_with_llm

# --- HELPER FUNCTION (Updated for nsusemiehl/SciERC structure) ---

def format_relations_for_comparison(sentences, ner_tags_per_sentence, relations_per_sentence):
    """
    Processes nsusemiehl/SciERC's sentence-based NER tags and relations into a set of tuples.
    """
    formatted_relations = set()
    current_token_index = 0
    entities_found = [] # Store entities as (start_token_idx, end_token_idx, label, text)

    # First pass: Extract all entities and their text spans based on NER tags (BIO format)
    for i, sentence in enumerate(sentences):
        ner_tags = ner_tags_per_sentence[i]
        in_entity = False
        start_idx = -1
        current_label = ""

        for j, token in enumerate(sentence):
            global_token_idx = current_token_index + j
            tag = ner_tags[j]

            if tag.startswith("B-"): # Begin entity
                if in_entity: # Close previous entity if open
                    entity_text = " ".join(sentence[start_idx:j]).lower()
                    entities_found.append( (current_token_index + start_idx, global_token_idx - 1, current_label, entity_text) )

                start_idx = j
                current_label = tag.split("-")[1] # Get label after "B-"
                in_entity = True
            elif tag.startswith("I-"): # Inside entity
                # Continue if inside an entity of the same type, otherwise handle potential errors/edge cases
                if not in_entity or tag.split("-")[1] != current_label:
                    if in_entity: # Close previous if mismatched I- tag
                        entity_text = " ".join(sentence[start_idx:j]).lower()
                        entities_found.append( (current_token_index + start_idx, global_token_idx - 1, current_label, entity_text) )
                    # Start a new entity based on I- tag (less ideal but handles some cases)
                    start_idx = j
                    current_label = tag.split("-")[1]
                    in_entity = True
            elif tag == "O": # Outside entity
                if in_entity: # Close previous entity
                    entity_text = " ".join(sentence[start_idx:j]).lower()
                    entities_found.append( (current_token_index + start_idx, global_token_idx - 1, current_label, entity_text) )
                in_entity = False
                start_idx = -1
                current_label = ""

        # Close any entity open at the end of the sentence
        if in_entity:
            entity_text = " ".join(sentence[start_idx:]).lower()
            entities_found.append( (current_token_index + start_idx, current_token_index + len(sentence) - 1, current_label, entity_text) )

        current_token_index += len(sentence) # Update global index offset

    # Second pass: Map relations to the extracted entity text
    current_token_index = 0
    for i, sentence_relations in enumerate(relations_per_sentence):
        sentence_len = len(sentences[i])
        for rel in sentence_relations:
            # Indices s1, e1, s2, e2 are relative to the start of the sentence
            s1, e1, s2, e2, label = rel
            source_text = None
            target_text = None

            # Find the entity text matching the relation spans (absolute indices)
            abs_s1 = current_token_index + s1
            abs_e1 = current_token_index + e1
            abs_s2 = current_token_index + s2
            abs_e2 = current_token_index + e2

            for ent_start, ent_end, ent_label, ent_text in entities_found:
                if ent_start == abs_s1 and ent_end == abs_e1:
                    source_text = ent_text
                if ent_start == abs_s2 and ent_end == abs_e2:
                    target_text = ent_text

            if source_text and target_text:
                formatted_relations.add((source_text, label.lower(), target_text))

        current_token_index += sentence_len # Update global index offset for next sentence

    return formatted_relations

# --- MAIN EVALUATION FUNCTION ---

def evaluate_knowledge_extraction():
    """
    Runs the evaluation on the nsusemiehl/SciERC dataset and prints performance metrics.
    """
    print("Loading SciERC dataset from Hugging Face (nsusemiehl/SciERC)...")
    try:
        # Load the specific dataset version we identified
        dataset = load_dataset("nsusemiehl/SciERC", split="test")
    except Exception as e:
        print(f"Failed to load dataset. Check your internet connection. Error: {e}")
        return

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    # For initial testing, you might want to process only a small subset
    # To run on the full dataset, remove the .select() part
    # subset_dataset = dataset.select(range(20)) # Use this for quick tests
    subset_dataset = dataset # Use this for the full run

    print(f"Starting evaluation on {len(subset_dataset)} documents...")

    # Added enumerate here to get the index for printing
    for item_index, item in enumerate(tqdm(subset_dataset, desc="Evaluating Documents")):
        # 1. Reconstruct the full text from the 'sentences' key
        try:
            sentences = item['sentences']
            full_text = " ".join([" ".join(sentence) for sentence in sentences])
        except KeyError:
            print(f"Skipping item {item_index + 1} - missing 'sentences' key.")
            continue # Skip this item if it doesn't have the expected structure

        # 2. Get the TRUE relations from the dataset using the specific keys
        try:
            true_relations = format_relations_for_comparison(
                item['sentences'], item['ner'], item['relations']
            )
        except KeyError as e:
            print(f"Skipping item {item_index + 1} - missing key for ground truth: {e}")
            continue # Skip if ground truth keys are missing

        # 3. Get the PREDICTED relations from our pipeline
        predicted_kg_data = extract_knowledge_with_llm(full_text)

        if predicted_kg_data and 'relationships' in predicted_kg_data:
            # Format the LLM output into the same tuple format
            predicted_relations_list = [
                (rel.get('source','').lower(), rel.get('type','').lower(), rel.get('target','').lower())
                for rel in predicted_kg_data['relationships']
            ]
            predicted_relations = set(predicted_relations_list)
        else:
            predicted_relations = set()

        # ---- START DEBUG PRINTS ----
        print(f"\nDocument {item_index + 1}:")
        print(f"  True relations found: {len(true_relations)}")
        # print(f"  Sample true: {list(true_relations)[:3]}") # Uncomment to see examples
        print(f"  Predicted relations found: {len(predicted_relations)}")
        # print(f"  Sample predicted: {list(predicted_relations)[:3]}") # Uncomment to see examples
        # ---- END DEBUG PRINTS ----

        # 4. Compare the sets to find TPs, FPs, and FNs
        true_positives = len(predicted_relations.intersection(true_relations))
        false_positives = len(predicted_relations.difference(true_relations))
        false_negatives = len(true_relations.difference(predicted_relations))

        # Accumulate the counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    # --- CALCULATE METRICS ---

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Knowledge Extraction Benchmark Results (nsusemiehl/SciERC) ---")
    print(f"Total True Relations in Dataset Subset: {total_true_positives + total_false_negatives}")
    print(f"Total Relations Extracted by LLM: {total_true_positives + total_false_positives}")
    print("---------------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("---------------------------------------------")
    print("Note: Precision = Of the relations extracted, how many were correct.")
    print("      Recall    = Of all correct relations, how many were found.")

if __name__ == "__main__":
    # Make sure .env is loaded before running evaluation
    load_dotenv()
    evaluate_knowledge_extraction()