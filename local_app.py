import json  # Add this import
import psutil
import torch
import boto3
from transformers import AutoTokenizer
import gradio as gr
import os
from typing import List, Tuple

class CustomerSupportBot:
    def __init__(self, endpoint_name="customer-support-gpt-2024-11-10-00-30-03-555"):
        self.process = psutil.Process(os.getpid())
        model_name = "EleutherAI/gpt-neo-125M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use the tokenizer appropriate to your model
        self.endpoint_name = endpoint_name
        self.sagemaker_runtime = boto3.client('runtime.sagemaker')
        
    def generate_response(self, message: str) -> str:
        try:
            input_text = f"Instruction: {message}\nResponse:"
            
            # Prepare payload for SageMaker endpoint
            payload = {
                # "inputs": inputs['input_ids'].tolist()[0],
                'inputs': input_text,
                # You can include other parameters if needed (e.g., attention_mask)
            }
            print(f'Payload: {payload}')
            # Convert the payload to a JSON string before sending
            json_payload = json.dumps(payload)  # Use json.dumps() to serialize the payload
            print(f'JSON Payload: {json_payload}')
            # Call the SageMaker endpoint for inference
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json_payload  # Send the JSON string here
            )
            print(f'Response: {response}')

            # Process the response
            result = response['Body'].read().decode('utf-8')
            parsed_result = json.loads(result)

            # Extract the generated text from the first element in the list
            generated_text = parsed_result[0]['generated_text']

            # Split the string to get the response part after 'Response:'
            response = generated_text.split('Response:')[1].strip()

            # return the extracted response
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
    bot = CustomerSupportBot()
    
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
        share=True,
        server_name="0.0.0.0",  # Makes the server accessible from other machines
        server_port=7860,  # Specify the port
        debug=True
    )
