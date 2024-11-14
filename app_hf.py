import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import os
import tarfile
from typing import List, Tuple
import boto3
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerSupportBot:
    def __init__(self, model_path=None):
        """
        Initialize the customer support bot with the fine-tuned model.
        
        Args:
            model_path (str): Path to the saved model and tokenizer
        """
        self.process = psutil.Process(os.getpid())

        if model_path is None:
            self.model_path = os.path.join(os.path.expanduser("~"), "customer_support_gpt")
        else:
            self.model_path = model_path


        self.model_path = Path(self.model_path)
        self.model_file_path = self.model_path / "model.tar.gz"

        self.s3 = boto3.client("s3")
        self.model_key = "models/model.tar.gz"
        self.bucket_name = "customer-support-gpt"
        
        # Download and load the model
        try:
            self.download_and_load_model()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def download_and_load_model(self):
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using model directory: {self.model_path}")

            # Download model from S3 if needed
            if not self.model_file_path.exists():
                logger.info("Downloading model from S3...")
                self.s3.download_file(self.bucket_name, self.model_key, str(self.model_file_path))
                logger.info("Download complete. Extracting model files...")

                # Extract the model files
                with tarfile.open(self.model_file_path, "r:gz") as tar:
                    tar.extractall(str(self.model_path))

            # Load the model and tokenizer
            logger.info("Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path))
            logger.info("Model and tokenizer loaded successfully.")

            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            logger.info(f'Model loaded on device: {self.device}')

        except PermissionError as e:
            logger.error(f"Permission error when accessing {self.model_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in download_and_load_model: {str(e)}")
            raise

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
    try:
        # Use a user-accessible directory for the model
        user_model_path = os.path.join(os.path.expanduser("~"), "customer_support_models")
        bot = CustomerSupportBot(model_path=user_model_path)
        
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
                elem_classes="message-box",
                # type="messages"
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

            print("Interface created successfully.")

            # call the initial query function
            # run a query first how are you and predict the output
            print(predict("How are you", []))

            # run a command which checks the resource usage
            print(f'Bot Resource Usage : {bot.monitor_resources()}')

            # show full system usage
            print(f'CPU Percentage : {psutil.cpu_percent()}')
            print(f'RAM Usage : {psutil.virtual_memory()}')
            print(f'Swap Memory : {psutil.swap_memory()}')

        return interface
        
    except Exception as e:
        logger.error(f"Failed to create chat interface: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting customer support bot application...")
        demo = create_chat_interface()
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True,
            inline=False
        )
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise