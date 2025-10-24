"""
model_loader.py
Handles loading and initialization of the Hugging Face model and tokenizer.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ModelLoader:
    """Loads and manages the text generation model and tokenizer."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the model loader with a specified model.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
        
    def load(self):
        """Load the model and tokenizer from Hugging Face."""
        print(f"Loading model: {self.model_name}...")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully!\n")
        
    def generate_response(self, input_ids, attention_mask, max_new_tokens=150):
        """
        Generate a response given input token IDs.
        
        Args:
            input_ids: Tokenized input tensor
            attention_mask: Attention mask tensor
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                temperature=0.7,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1
            )
        return output
    
    def get_tokenizer(self):
        """Return the tokenizer instance."""
        return self.tokenizer
    
    def get_model(self):
        """Return the model instance."""
        return self.model