from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

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
                max_length=50,
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

# Initialize the chatbot
bot = CustomerSupportBot()

# Define the Gradio interface function
def chatbot_response(message, history):
    """
    Generate bot response for the Gradio interface.
    
    Args:
        message (str): User's input message
        history (list): Chat history
    """
    bot_response = bot.generate_response(message)
    history.append((bot_response))
    return history

# Create the Gradio interface
iface = gr.ChatInterface(
    fn=chatbot_response,
    title="Customer Support Chatbot",
    description="Ask your questions to the customer support bot!",
    examples=["How do I reset my password?", 
             "What are your shipping policies?", 
             "I want to return a product."],
    # retry_btn=None,
    # undo_btn="Remove Last",
    # clear_btn="Clear",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=False)  # Set share=True if you want to create a public link