from google import genai
import torch, os
from dotenv import load_dotenv

class GeminiLLMClient:
    """
    Client wrapper for interacting with the Gemini LLM.

    Handles authentication and text generation using a configured Gemini model.
    """
    def __init__(self,
                 model_name: str = 'gemini-2.5-flash'
                 ):        
        self.api_key = os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    
    def generate(self, prompt: str) -> str:
        """
        Generates a text response from the Gemini model.

        Args:
            prompt: Input prompt provided to the language model.

        Returns:
            Generated text response.
        """
        refined_response = self.client.models.generate_content(
            model=self.model_name,
            contents= prompt
        )
        return refined_response.text
    

