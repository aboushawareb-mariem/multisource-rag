from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
import torch, os

class LLMClient:
    """
    Proof-of-concept local LLM client using open-source transformer models.

    This implementation is provided for experimentation and future extension.
    In practice, smaller open-source models (e.g. TinyLlama, Phi-3 mini)
    were found to produce inconsistent and low-quality refinements for this task.

    As a result, this client is not used by default in the pipeline.
    A hosted LLM (e.g. Gemini) is preferred for reliable answer refinement.
    """
    def __init__(self,
                #  model_name: str = 'microsoft/phi-3.5-mini-instruct',
                 model_name: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                 device: str = 'cpu',
                 max_new_tokens: int = 200,
                 ):
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)

        self.device = device
        self.max_new_tokens = max_new_tokens
    
    def generate(self, prompt: str) -> str:
        """
        Generates a response using a local causal language model.

        Note:
            Output quality may be limited due to model size and lack of instruction
            tuning compared to hosted LLMs.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample = False
            )

        refined = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        refined, prompt = normalize(refined), normalize(prompt)

        # For causal LMs, output often includes the prompt; strip it.
        if refined.startswith(prompt):
            refined = refined[len(prompt):]
        return refined
    
def normalize(text: str) -> str:
    return " ".join(text.split())
