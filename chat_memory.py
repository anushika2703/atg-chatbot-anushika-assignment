"""
chat_memory.py
Implements a sliding window memory buffer for maintaining conversation context.
"""

import torch


class ChatMemory:
    """Manages conversation history using a sliding window approach."""
    
    def __init__(self, window_size=5, tokenizer=None, is_chat_model=False):
        """
        Initialize the chat memory buffer.
        
        Args:
            window_size (int): Number of recent exchanges to remember
            tokenizer: Hugging Face tokenizer instance
            is_chat_model (bool): Whether using a chat-tuned model
        """
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.is_chat_model = is_chat_model
        self.history = []  # List of (user_msg, bot_msg) tuples
        
    def add_exchange(self, user_msg, bot_msg):
        """
        Add a user-bot exchange to memory.
        
        Args:
            user_msg (str): User's message
            bot_msg (str): Bot's response
        """
        self.history.append((user_msg, bot_msg))
        
        # Maintain sliding window by removing oldest exchange if needed
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_context(self):
        """
        Get the current conversation context as a formatted string.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.history:
            return ""
        
        if self.is_chat_model:
            # Format for chat models with system/user/assistant structure
            context = ""
            for user_msg, bot_msg in self.history:
                context += f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{bot_msg}</s>\n"
            return context
        else:
            # Simple format for base models
            context = ""
            for user_msg, bot_msg in self.history:
                context += f"User: {user_msg}\nBot: {bot_msg}\n"
            return context
    
    def format_prompt(self, user_input):
        """
        Format the prompt with history for chat models.
        
        Args:
            user_input (str): Current user message
            
        Returns:
            str: Formatted prompt
        """
        if self.is_chat_model:
            # TinyLlama chat format
            prompt = "<|system|>\nYou are a helpful AI assistant. Answer questions accurately and concisely. Maintain context from previous messages in the conversation.</s>\n"
            
            # Add history
            for user_msg, bot_msg in self.history:
                prompt += f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{bot_msg}</s>\n"
            
            # Add current user input
            prompt += f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
        else:
            # Simple format for base models
            prompt = ""
            for user_msg, bot_msg in self.history:
                prompt += f"User: {user_msg}\nBot: {bot_msg}\n"
            prompt += f"User: {user_input}\nBot:"
        
        return prompt
    
    def clear(self):
        """Clear all conversation history."""
        self.history = []
        print("Memory cleared.\n")
    
    def get_memory_size(self):
        """
        Get the current number of exchanges in memory.
        
        Returns:
            int: Number of exchanges stored
        """
        return len(self.history)