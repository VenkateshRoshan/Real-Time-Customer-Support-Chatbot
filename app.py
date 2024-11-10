import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import os
import tarfile
from typing import List, Tuple
import boto3

class CustomerSupportBot:
    def __init__(self, model_path="models/customer_support_gpt"):
        """
        Initialize the customer support bot with the fine-tuned model.
        
        Args:
            model_path (str): Path to the saved model and tokenizer
        """
        self.process = psutil.Process(os.getpid())
        self.model_path = model_path
        self.model_file_path = os.path.join(self.model_path, "model.tar.gz")
        self.s3 = boto3.client("s3")
        self.model_key = "models/model.tar.gz"
        self.bucket_name = "customer-support-gpt"
        
        # Download and load the model
        self.download_and_load_model()

    def download_and_load_model(self):
        # Check if the model directory exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Download model.tar.gz from S3 if not already downloaded
        if not os.path.exists(self.model_file_path):
            print("Downloading model from S3...")
            self.s3.download_file(self.bucket_name, self.model_key, self.model_file_path)
            print("Download complete. Extracting model files...")

            # Extract the model files
            with tarfile.open(self.model_file_path, "r:gz") as tar:
                tar.extractall(self.model_path)

        # Load the model and tokenizer from extracted files
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        print("Model and tokenizer loaded successfully.")

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def generate_response(self, message: str, max_length=100, temperature=0.7) -> str:
        try:
            input_text = f"Instruction: {message}\nResponse:"
            
            # Tokenize input text
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Generate response using the model
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
            
            # Decode and format the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Response:")[-1].strip()
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def monitor_resources(self) -> dict:
        usage = {
            "CPU (%)": self.process.cpu_percent(interval=1),
            "RAM (GB)": self.process.memory_info().rss / (1024 ** 3)
        }
        return usage


def create_chat_interface():
    bot = CustomerSupportBot(model_path="/app/models")
    
    def predict(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        if not message:
            return "", history
        
        bot_response = bot.generate_response(message)
        
        # Log resource usage
        usage = bot.monitor_resources()
        print("Resource Usage:", usage)
        
        history.append((message, bot_response))
        return "", history

    # Create the Gradio interface with custom CSS
    with gr.Blocks(css="""
        .message-box {
            margin-bottom: 10px;
        }
        .button-row {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
    """) as interface:
        gr.Markdown("# Customer Support Chatbot")
        gr.Markdown("Welcome! How can I assist you today?")
        
        chatbot = gr.Chatbot(
            label="Chat History",
            height=500,
            elem_classes="message-box"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2,
                elem_classes="message-box"
            )
        
        with gr.Row(elem_classes="button-row"):
            submit = gr.Button("Send Message", variant="primary")
            clear = gr.ClearButton([msg, chatbot], value="Clear Chat")

        # Add example queries in a separate row
        with gr.Row():
            gr.Examples(
                examples=[
                    "How do I reset my password?",
                    "What are your shipping policies?",
                    "I want to return a product.",
                    "How can I track my order?",
                    "What payment methods do you accept?"
                ],
                inputs=msg,
                label="Example Questions"
            )

        # Set up event handlers
        submit_click = submit.click(
            predict,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            predict,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        # Add keyboard shortcut for submit
        msg.change(lambda x: gr.update(interactive=bool(x.strip())), inputs=[msg], outputs=[submit])

    return interface

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Makes the server accessible from other machines
        server_port=7860,  # Specify the port
        debug=True
    )
