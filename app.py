import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os
from typing import List, Tuple

class CustomerSupportBot:
    def __init__(self, model_path="models/customer_support_gpt"):
        self.process = psutil.Process(os.getpid())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def generate_response(self, message: str) -> str:
        try:
            input_text = f"Instruction: {message}\nResponse:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Response:")[-1].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def monitor_resources(self) -> dict:
        usage = {
            "CPU (%)": self.process.cpu_percent(interval=1),
            "RAM (GB)": self.process.memory_info().rss / (1024 ** 3)
        }
        if torch.cuda.is_available():
            usage["GPU (GB)"] = torch.cuda.memory_allocated(0) / (1024 ** 3)
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
            height=400,
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