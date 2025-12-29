from models.llm_client import LLMClient
from rag.video_answer import VideoAnswer
from rag.pdf_answer import PDFAnswer


VIDEO_PROMPT_TEMPLATE = """You are a helpful editor.
Correct the grammar and wording of the answer/ instruction.
Fix the grammar and make the answer/ instruction sound natural.
Do not add or remove any information.
Do NOT explain your reasoning.
Do NOT include thoughts or analysis.
Return ONLY the corrected instruction.
Return ONLY the corrected instruction. Do NOT explain.
If the sentence already makes sense, then do not change it.
keep the answer concise and dont move on to unrelated subjects


Here is an input example:

Question:
"How do I add a new customer?"
Sentence:
"a new customer click on the customers tab Then press"

output example:
"To add a new customer, click on the customer tab then press"

Question:
{QUESTION}
Sentence:
{TEXT}

output:
"""

PDF_PROMPT_TEMPLATE = """ You are a helpful editor. Summarize the following answer for this question:
question:
{QUESTION}
answer:
{TEXT}

summary:

"""



class AnswerRefiner:
    """
    Uses an LLM to improve or summarize retrieved answers.

    This class post-processes answers retrieved from the RAG pipeline by:
    - Rewriting video transcript snippets to be more natural and readable.
    - Summarizing PDF-based answers.

    The refinement is performed in-place by mutating the provided
    VideoAnswer or PDFAnswer objects.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def refine_answer(self, question: str, video_answer: VideoAnswer | None = None, pdf_answer: PDFAnswer | None = None) -> str:
        """
        Refines retrieved answers using the language model.

        If a video answer is provided, its transcript snippet is rewritten
        to improve grammar and clarity without altering the original meaning.

        If a PDF answer is provided, its text is summarized with respect to
        the user question.

        The refinement is applied in-place by setting:
        - `video_answer.refined_answer`
        - `pdf_answer.summary`

        Args:
            question: The original user question.
            video_answer: Retrieved video-based answer, if available.
            pdf_answer: Retrieved PDF-based answer, if available.

        Returns:
            A tuple containing the updated (video_answer, pdf_answer).
        """
        
        if (video_answer):
            answer_prompt = VIDEO_PROMPT_TEMPLATE.format(QUESTION = question, TEXT = video_answer.transcript_snippet)
            refined_answer = self.llm_client.generate(answer_prompt)
            video_answer.refined_answer = refined_answer
        
        if(pdf_answer):
            answer_prompt = PDF_PROMPT_TEMPLATE.format(QUESTION = question, TEXT = pdf_answer.text)
            summary = self.llm_client.generate(answer_prompt)
            pdf_answer.summary = summary 
        
        return video_answer, pdf_answer


        