from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CustomerSupportBot:
    def __init__(self, model_path="models/customer_support_gpt"):
        """
        Initialize the customer support bot with the fine-tuned model.
        
        Args:
            model_path (str): Path to the saved model and tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def generate_response(self, instruction, max_length=100, temperature=0.7):
        """
        Generate a response for a given customer support instruction/query.
        
        Args:
            instruction (str): Customer's query or instruction
            max_length (int): Maximum length of the generated response
            temperature (float): Controls randomness in generation (higher = more random)
            
        Returns:
            str: Generated response
        """
        # Format input text the same way as during training
        input_text = f"Instruction: {instruction}\nResponse:"
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )
        
        # Decode and format response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        response = response.split("Response:")[-1].strip()
        
        return response

def main():
    # Initialize the bot
    bot = CustomerSupportBot()
    
    # Example queries
    example_queries = [
        "How do I reset my password?",
        "What are your shipping policies?",
        "I want to return a product.",
    ]
    
    # Generate and print responses
    print("Customer Support Bot Demo:\n")
    for query in example_queries:
        print(f"Customer: {query}")
        response = bot.generate_response(query)
        print(f"Bot: {response}\n")

    # Interactive mode
    print("Enter your questions (type 'quit' to exit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'quit':
            break
            
        response = bot.generate_response(query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()