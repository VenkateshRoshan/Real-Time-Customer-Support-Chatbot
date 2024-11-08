import pandas as pd
from transformers import AutoTokenizer

def load_data(file_path):
    """
    Load the customer support dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess data by tokenizing the instructions and responses.
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_data(row):
        """
        Helper function to tokenize instruction and response.
        """
        instruction_tokens = tokenizer(row['instruction'], truncation=True, padding="max_length", max_length=256)
        response_tokens = tokenizer(row['response'], truncation=True, padding="max_length", max_length=256)
        return instruction_tokens, response_tokens

    # Tokenize each row's instruction and response
    data['instruction_tokens'], data['response_tokens'] = zip(*data.apply(tokenize_data, axis=1))
    return data[['instruction_tokens', 'response_tokens']]

if __name__ == "__main__":
    data = load_data('data/raw/customer_support.csv')
    processed_data = preprocess_data(data)
    processed_data.to_csv('data/processed/customer_support_preprocessed.csv', index=False)
