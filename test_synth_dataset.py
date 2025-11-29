"""
Quick test to inspect SYNTH dataset structure
"""

from datasets import load_dataset

print("Loading SYNTH dataset...")
print("This might take a moment...\n")

# Load dataset in streaming mode
dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)

print("Dataset loaded successfully!")
print(f"Dataset type: {type(dataset)}")
print()

# Get first example
print("Fetching first example...")
first_example = next(iter(dataset))

print("\n" + "="*60)
print("DATASET STRUCTURE")
print("="*60)
print(f"\nKeys in each example: {list(first_example.keys())}")
print()

# Print each field
for key, value in first_example.items():
    print(f"\n{key}:")
    print(f"  Type: {type(value)}")
    if isinstance(value, str):
        print(f"  Length: {len(value)} characters")
        print(f"  Preview: {value[:200]}...")
    else:
        print(f"  Value: {value}")

print("\n" + "="*60)
print("SAMPLE TEXTS FROM DATASET")
print("="*60)

# Show a few more examples
for i, example in enumerate(dataset):
    if i >= 3:
        break
    print(f"\nExample {i+1}:")
    if 'text' in example:
        print(example['text'][:300] + "...")
    elif 'content' in example:
        print(example['content'][:300] + "...")
    else:
        print(example)
