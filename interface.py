"""
interface.py
Command-line interface for the chatbot with conversation loop.
"""

from model_loader import ModelLoader
from chat_memory import ChatMemory


class ChatInterface:
    """Command-line interface for the conversational chatbot."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", memory_window=5):
        """
        Initialize the chat interface.
        
        Args:
            model_name (str): Hugging Face model to use
            memory_window (int): Number of exchanges to remember
        """
        self.model_loader = ModelLoader(model_name)
        self.model_loader.load()
        
        self.memory = ChatMemory(
            window_size=memory_window,
            tokenizer=self.model_loader.get_tokenizer(),
            is_chat_model=self.model_loader.is_chat_model
        )
        
        self.tokenizer = self.model_loader.get_tokenizer()
        self.device = self.model_loader.device
        
    def display_welcome(self):
        """Display welcome message and instructions."""
        print("=" * 60)
        print("Welcome to the Chatbot!")
        print("=" * 60)
        print("Commands:")
        print("  /exit    - Exit the chatbot")
        print("  /clear   - Clear conversation memory")
        print("  /memory  - Show current memory usage")
        print("=" * 60)
        print()
    
    def process_command(self, user_input):
        """
        Process special commands.
        
        Args:
            user_input (str): User's input
            
        Returns:
            bool: True if command was processed, False otherwise
        """
        command = user_input.strip().lower()
        
        if command == "/exit":
            print("\nExiting chatbot. Goodbye!")
            return True
        elif command == "/clear":
            self.memory.clear()
            return False
        elif command == "/memory":
            size = self.memory.get_memory_size()
            print(f"Memory: {size}/{self.memory.window_size} exchanges stored\n")
            return False
        
        return False
    
    def generate_reply(self, user_input):
        """
        Generate a bot reply to user input.
        
        Args:
            user_input (str): User's message
            
        Returns:
            str: Bot's response
        """
        # Format prompt with conversation history
        prompt = self.memory.format_prompt(user_input)
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate response
        output_ids = self.model_loader.generate_response(
            inputs['input_ids'],
            inputs['attention_mask']
        )
        
        # Decode the full output
        full_response = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Extract only the bot's reply (everything after the last assistant marker)
        if self.memory.is_chat_model:
            # For chat models, extract text after the last <|assistant|> tag
            if "<|assistant|>" in full_response:
                bot_reply = full_response.split("<|assistant|>")[-1].strip()
            else:
                bot_reply = full_response[len(prompt):].strip()
        else:
            # For base models, extract text after "Bot:"
            bot_reply = full_response[len(prompt):].strip()
        
        # Clean up the reply
        bot_reply = bot_reply.split("<|user|>")[0].strip()  # Remove any user prompts
        bot_reply = bot_reply.split("\n\n")[0].strip()  # Take only first paragraph
        
        # Store exchange in memory
        self.memory.add_exchange(user_input, bot_reply)
        
        return bot_reply
    
    def run(self):
        """Main conversation loop."""
        self.display_welcome()
        
        try:
            while True:
                # Get user input
                user_input = input("User: ").strip()
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.startswith("/"):
                    should_exit = self.process_command(user_input)
                    if should_exit:
                        break
                    continue
                
                # Generate and display bot response
                bot_reply = self.generate_reply(user_input)
                print(f"Bot: {bot_reply}")
                print()
                
        except KeyboardInterrupt:
            print("\n\nExiting chatbot. Goodbye!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Exiting chatbot. Goodbye!")


def main():
    """Entry point for the chatbot application."""
    # You can customize the model and memory window size here
    chatbot = ChatInterface(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Better for Q&A
        memory_window=5  # Remember last 5 exchanges
    )
    chatbot.run()


if __name__ == "__main__":
    main()