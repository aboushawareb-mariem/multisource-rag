def format_answer(question, video_answer=None, pdf_answer=None) -> str:
    """
    Formats the final answer for presentation to the user.

    Combines the user question with the retrieved answer source
    (video or PDF) and includes both the raw and refined outputs
    when available.

    Args:
        question: The original user question.
        video_answer: A VideoAnswer object containing video-based results.
        pdf_answer: A PDFAnswer object containing PDF-based results.

    Returns:
        A formatted multi-line string suitable for display in CLI
        output or API responses.
    """
    
    lines = []
    lines.append("\n=== QUESTION ===")
    lines.append(question)

    lines.append("\n=== SOURCE ===")

    if video_answer:
        lines.append("Type: Video")
        lines.append(f"Video ID: {video_answer.video_id}")
        lines.append(
            f"Timestamps: {video_answer.start_timestamp} – {video_answer.end_timestamp}"
        )
        lines.append(
            f"Start token_id: {video_answer.start_token_id} End token_id: {video_answer.end_token_id}"
        )
        lines.append("\n=== RAW ANSWER ===")
        lines.append(video_answer.transcript_snippet)

        lines.append("\n=== REFINED ANSWER ===")
        lines.append(video_answer.refined_answer)

    elif pdf_answer:
        lines.append("Type: PDF")
        lines.append(f"PDF: {pdf_answer.pdf_id}")
        lines.append(f"Page: {pdf_answer.page_number}")
        lines.append(f"Paragraph: {pdf_answer.paragraph_index}")
        lines.append(f"raw text: {pdf_answer.text}")
        lines.append(f"summary: {pdf_answer.summary}")
    else:
        lines.append("No answer was available due to lack of resources. Please provide more resources and try again.")

    final = "\n".join(lines)
    return "\n".join(lines)
