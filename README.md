# Explaino RAG System
This project implements a Retrieval-Augmented Generation (RAG) system that answers user questions by retrieving relevant information from video transcripts and PDF documents, then refining the response using a language model.

The system is designed to be extensible, configurable, and easy to reason about, with clear separation between preprocessing, retrieval, indexing, and answer refinement.

This can be used as a first draft for the system design, then the logic (for chunking and indexing for example) can be extended easily from this architecture.

## File Structure

```
.
├── api                         
│   └── app.py                      # FastAPI
├── data
│   ├── pdf_source                          # dummy sample pdf data
│   │   ├── user_manual_1.pdf
│   │   └── user_manual_2.pdf
│   └── video_transcript_source             # dummy sample video transcripts data
│       ├── video_001.json
│       ├── video_002.json
│       ├── video_003.json
│       ├── video_004.json
│       └── video_005.json
├── dummy.py
├── experiments.ipynb
├── indexing
│   └── faiss
│       └── vector_index.py
├── main.py                         # full orchestration from preprocessing to answer reformatting
├── models
│   ├── embedder.py                         # embedding language model
│   ├── gemini_llm_client.py                # gemini-like llms client
│   └── llm_client.py                       # opensource llms client (no api key)
├── preprocessing                                     
│   ├── pdf                             # pdf source loading and chunking
│   │   ├── load_pdfs_data.py
│   │   ├── pdf_chunk.py
│   │   └── pdf_chunking.py
│   └── video                           # video source loading and chunking
│       ├── load_videos_data.py
│       ├── transcript_chunk.py
│       └── video_chunking.py
├── rag                             # orchestration of the RAG system (indexing, reformatting)
│   ├── answer_refiner.py
│   ├── format_answers.py
│   ├── pdf_answer.py
│   ├── rag_system.py
│   ├── retrievel.py
│   └── video_answer.py
├── README.md
└── tests                       # test suite
    ├── conftest.py
    ├── test_answer_refiner.py
    ├── test_format_answer.py
    ├── test_pdf_chunking.py
    ├── test_retrieval_smoke.py
    ├── test_vector_index.py
    └── test_video_chunking.py
```

## Features
### Video preprocessing and chunking:
- Load all video transcript json files.
- Create video chunks that span tokens of size chunk_size. An overlap between the chunks can also be specified.
### PDF preprocessing and chunking:
- Loading all pdf files in the directory.
- I used fitz/ pymupdf library for the pdf processing.
- One of the main features of this library is working with pages. It parses the pdf as a list of pages and gives the page number as the index of the page in the list.
- Chunking pdfs - unlike videos - relies more on chunking paragraphs together.
### Embedding:
- Embedding model is configurable through environment variables.
- Used a default embedding model that is well suited for information retrieval and similarity tasks: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Using the model in a wrapper in order to create a high-level orchestration of the model and separation of concerns. This allows to extend it offline at any point if needed.
- Used to embed video and pdf chunks.
### Vector DB:
- Used FAISS: Memory efficient, fast, scalable, allows different modes of indexing, easy to setup.
- For the purpose of this task, a restart of the program (for example if running through the API) is needed if extra data is added for the db to update, however this can be easily extended later since everything is modularized.
- FAISS is also used in a wrapper under the directory `indexing.faiss` in order to easily allow for adding other indexing options or extend this one internally.
### Rag orchestration:
- The rag orchestration consists of: retrieval, answer construction and refining.
- RagSystem uses the embedder and retrieves the best match from the vector index. 
- Retrieval is split between from video source or pdf source.
### main
- Implements video -> pdf -> default no answer response logic.
- Orchestrates the system from all the different wrappers.
-- video: Load, retriever (chunk, embed, and retrieve)
-- If no answer from video, repeat for pdf
-- Refine answer with LLMClient
-- Format and print answer.
## How To?
- Upload data in the `data` folder either in `pdf_source` or `video_transcript_source` depending on the type.
- Running with the terminal: `python3 main.py --question "<your question>"`
- Running with FastAPI: start the server `uvicorn api.app:app --reload` then go to `localhost:8000/docs` go to /ask endpoint and edit the request json.
- Example queestion for video source: "How to find a list of existing customers?"
- Example question for pdf source: "what is mojo 2?"
- Configurations such as paths, api keys, model names, thresholds can be tweaked through environment variables.
    - Example for the .env file is found in .env.example
## Dependencies
- sentence-transformers
- faiss-cpu
- numpy
- fitz/ PyMuPDF
- google-generativeai
- fastapi
- pytest
- python-dotenv
- uvicorn

## Limitations & Future Work
- Indexes are rebuilt on restart; incremental indexing is not yet supported
- Retrieval currently selects the top match; top-k aggregation could improve recall
- Answer refinement quality depends on the external LLM
- The threshold might be too harsh for the pdf source since it has more tokens than in the video because it's paragraph based.

#### Notes on using this repo: This project is provided for evaluation purposes only. Commercial use is not permitted without explicit permission.

