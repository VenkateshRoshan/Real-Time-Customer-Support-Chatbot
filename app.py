import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import os
import tarfile
from typing import List, Tuple
import boto3
from flask import Flask, Response, jsonify, request
import threading
import psutil
import logging
from waitress import serve

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthCheckServer:
    def __init__(self, bot=None):
        self.app = Flask(__name__)
        self.bot = bot
        
        # Health check endpoint
        @self.app.route("/ping", methods=["GET"])
        def ping():
            try:
                # Check system health
                healthy = self.check_system_health()
                if healthy:
                    logger.info("Health check passed")
                    return Response(response='\n', status=200, mimetype='application/json')
                else:
                    logger.error("Health check failed")
                    return Response(response='\n', status=500, mimetype='application/json')
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                return Response(response='\n', status=500, mimetype='application/json')

        # Inference endpoint
        @self.app.route("/invocations", methods=["POST"])
        def invocations():
            try:
                if not request.is_json:
                    return Response(response='This predictor only supports JSON data',
                                  status=415,
                                  mimetype='text/plain')
                
                data = request.get_json()
                message = data.get('message', '')
                
                if not message:
                    return Response(response=json.dumps({"error": "No message provided"}),
                                  status=400,
                                  mimetype='application/json')
                
                response = self.bot.generate_response(message)
                return Response(response=json.dumps({"response": response}),
                              status=200,
                              mimetype='application/json')
                
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                return Response(response=json.dumps({"error": str(e)}),
                              status=500,
                              mimetype='application/json')
    
    def check_system_health(self):
        """Check if system and model are healthy"""
        try:
            # Check if model is loaded
            if self.bot and not hasattr(self.bot, 'model'):
                logger.error("Model not loaded")
                return False
            
            # Check memory usage
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                logger.error(f"High memory usage: {mem.percent}%")
                return False
            
            # Check CPU usage
            if psutil.cpu_percent() > 95:
                logger.error(f"High CPU usage: {psutil.cpu_percent()}%")
                return False
            
            # Log current resource usage
            logger.info(f"System health: Memory {mem.percent}%, CPU {psutil.cpu_percent()}%")
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
    
    def run(self):
        """Run the health check server"""
        logger.info("Starting health check server on port 8080...")
        serve(self.app, host='0.0.0.0', port=8080)

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
        self.device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        print(f'Model loaded on device: {self.device}')

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
    
    # Start health check server
    health_server = HealthCheckServer(bot)
    health_thread = threading.Thread(target=health_server.run, daemon=True)
    health_thread.start()

    # Function to run initial query
    def initial_query():
        welcome_message = "Hello! I'm your customer support assistant. How can I help you today?"
        return "", [(None, welcome_message)]
    
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

if __name__ == "__main__":
    demo = create_chat_interface()
    print("Starting Gradio server...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,  # Changed to 7860 for Gradio
        debug=True,
        inline=False
    )
