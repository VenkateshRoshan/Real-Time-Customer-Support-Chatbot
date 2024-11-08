from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("models/customer_support_gpt")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

class Query(BaseModel):
    query: str

@app.post("/predict/")
def predict(query: Query):
    inputs = tokenizer(query.query, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
