from datasets import load_dataset
import pandas as pd

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
# save the dataset to a pandas dataframe only the instruction and response features
df = pd.DataFrame(ds['train'])
df = df[['instruction', 'response']]

# save the dataframe to a csv file
df.to_csv('data/raw/customer_support.csv', index=False)