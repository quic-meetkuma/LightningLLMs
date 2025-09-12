
from datasets import load_dataset
train_data = load_dataset("knkarthick/samsum", split="train")
test_data = load_dataset("knkarthick/samsum", split="test")

# Filter out empty samples
def is_not_empty(example):
    if example['dialogue'] is None or example['summary'] is None:
        return False
    return bool(example['dialogue'].strip()) and bool(example['summary'].strip())

train_data = train_data.filter(is_not_empty)
test_data = test_data.filter(is_not_empty)

def preprocess_function(example):
    return {
        # "prompt": example['dialogue'],
        "prompt": f"Summarize this dialog:\n{example['dialogue']}\n---\nSummary:\n",
        "completion": example['summary'],
    }

train_data = train_data.map(preprocess_function, remove_columns=["id", "dialogue", "summary"])
test_data = test_data.map(preprocess_function, remove_columns=["id", "dialogue", "summary"])
