from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import psutil
import os
import time
from typing import Dict, Any
import numpy as np

class MemoryTracker:
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            'gpu': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0  # GPU memory in MB
        }
    
    @staticmethod
    def format_memory_stats(stats: Dict[str, float]) -> str:
        """Format memory statistics into a readable string."""
        return (f"RSS Memory: {stats['rss']:.2f} MB\n"
                f"Virtual Memory: {stats['vms']:.2f} MB\n"
                f"GPU Memory: {stats['gpu']:.2f} MB")

class CustomerSupportBot:
    def __init__(self, model_path="models/customer_support_gpt"):
        """
        Initialize the customer support bot with the fine-tuned model and memory tracking.
        
        Args:
            model_path (str): Path to the saved model and tokenizer
        """
        # Record initial memory state
        self.initial_memory = MemoryTracker.get_memory_usage()
        
        # Load tokenizer and track memory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.post_tokenizer_memory = MemoryTracker.get_memory_usage()
        
        # Load model and track memory
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.post_model_memory = MemoryTracker.get_memory_usage()
        
        # Move model to GPU if available
        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.post_device_memory = MemoryTracker.get_memory_usage()
        
        # Calculate memory deltas
        self.memory_deltas = {
            'tokenizer_load': {k: self.post_tokenizer_memory[k] - self.initial_memory[k] 
                             for k in self.initial_memory},
            'model_load': {k: self.post_model_memory[k] - self.post_tokenizer_memory[k] 
                          for k in self.initial_memory},
            'device_transfer': {k: self.post_device_memory[k] - self.post_model_memory[k] 
                              for k in self.initial_memory}
        }
        
        # Initialize inference memory tracking
        self.inference_memory_stats = []
    
    def get_memory_report(self) -> str:
        """Generate a comprehensive memory usage report."""
        report = ["Memory Usage Report:"]
        
        report.append("\nModel Loading Memory Changes:")
        report.append("Tokenizer Loading:")
        report.append(MemoryTracker.format_memory_stats(self.memory_deltas['tokenizer_load']))
        
        report.append("\nModel Loading:")
        report.append(MemoryTracker.format_memory_stats(self.memory_deltas['model_load']))
        
        report.append("\nDevice Transfer:")
        report.append(MemoryTracker.format_memory_stats(self.memory_deltas['device_transfer']))
        
        if self.inference_memory_stats:
            avg_inference_memory = {
                k: np.mean([stats[k] for stats in self.inference_memory_stats])
                for k in self.inference_memory_stats[0]
            }
            report.append("\nAverage Inference Memory Usage:")
            report.append(MemoryTracker.format_memory_stats(avg_inference_memory))
        
        return "\n".join(report)
        
    def generate_response(self, instruction, max_length=100, temperature=0.7):
        """
        Generate a response for a given customer support instruction/query with memory tracking.
        
        Args:
            instruction (str): Customer's query or instruction
            max_length (int): Maximum length of the generated response
            temperature (float): Controls randomness in generation
            
        Returns:
            tuple: (Generated response, Memory usage statistics)
        """
        # Record pre-inference memory
        pre_inference_memory = MemoryTracker.get_memory_usage()
        
        # Format and tokenize input
        input_text = f"Instruction: {instruction}\nResponse:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate response and track memory
        start_time = time.time()
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
        inference_time = time.time() - start_time
        
        # Record post-inference memory
        post_inference_memory = MemoryTracker.get_memory_usage()
        
        # Calculate memory delta for this inference
        inference_memory_delta = {
            k: post_inference_memory[k] - pre_inference_memory[k]
            for k in pre_inference_memory
        }
        self.inference_memory_stats.append(inference_memory_delta)
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Response:")[-1].strip()
        
        return response, {
            'memory_delta': inference_memory_delta,
            'inference_time': inference_time
        }

def main():
    # Initialize the bot
    print("Initializing bot and tracking memory usage...")
    bot = CustomerSupportBot()
    print(bot.get_memory_report())
    
    # Example queries
    example_queries = [
        "How do I reset my password?",
        "What are your shipping policies?",
        "I want to return a product.",
    ]
    
    # Generate and print responses with memory stats
    print("\nCustomer Support Bot Demo:\n")
    for query in example_queries:
        print(f"Customer: {query}")
        response, stats = bot.generate_response(query)
        print(f"Bot: {response}")
        print(f"Inference Memory Delta: {MemoryTracker.format_memory_stats(stats['memory_delta'])}")
        print(f"Inference Time: {stats['inference_time']:.2f} seconds\n")
    
    # Interactive mode
    print("Enter your questions (type 'quit' to exit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'quit':
            break
        
        response, stats = bot.generate_response(query)
        print(f"Bot: {response}")
        print(f"Inference Memory Delta: {MemoryTracker.format_memory_stats(stats['memory_delta'])}")
        print(f"Inference Time: {stats['inference_time']:.2f} seconds")
    
    # Print final memory report
    print("\nFinal Memory Report:")
    print(bot.get_memory_report())

if __name__ == "__main__":
    main()