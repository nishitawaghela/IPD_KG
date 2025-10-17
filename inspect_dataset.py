# inspect_dataset.py
from datasets import load_dataset

print("Loading one item from hrithikpiyush/scierc...")
try:
    # Load just one item from the test split
    dataset = load_dataset("hrithikpiyush/scierc", split="test")
    first_item = dataset[0] # Get the very first document

    # Print all the available keys for this item
    print("\nAvailable keys in the first item:")
    print(list(first_item.keys()))

    # Let's also print the first 50 characters of likely text fields
    if 'text' in first_item:
         print("\nPreview of 'text':", first_item['text'][:50] + "...")
    if 'sentences' in first_item:
         print("\nPreview of 'sentences' (first sentence):", first_item['sentences'][0][:10] + "...") # Show first 10 words of first sentence

except Exception as e:
    print(f"\nAn error occurred: {e}")