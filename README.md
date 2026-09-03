# YouTube AI Assistant

An AI-powered YouTube video question-answering application built using **Retrieval-Augmented Generation (RAG)**.

YouTube AI Assistant allows users to provide a YouTube video URL and ask questions about its content. The application extracts the video's transcript, processes it into chunks, creates embeddings, retrieves the most relevant information using FAISS, and generates context-aware answers using Mistral AI.

---

## Navigation

- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [RAG Pipeline](#rag-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Example Questions](#example-questions)
- [Use Cases](#use-cases)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Security](#security)
- [Author](#author)

---

## Project Preview

<p align="center">
  <img src="assets/screenshot.png" alt="YouTube AI Assistant" width="900">
</p>

---

## Live Demo

Try the application here:

**Live Demo:**  
https://readmemd-pnt3rj954vhyu8efywognn.streamlit.app/

---

## GitHub Repository

**GitHub:**  
https://github.com/jatin-kumar210/YouTube-AI-Assistant

---

## Overview

Watching long YouTube videos to find specific information can be time-consuming.

YouTube AI Assistant solves this problem by allowing users to interact with a video's transcript using natural language.

Instead of manually searching through a long video, users can simply ask questions such as:

> "What is the main topic discussed in this video?"

or

> "Explain the concept mentioned in the second half of the video."

The system retrieves the most relevant transcript sections and uses them as context for the language model.

This makes the answers more relevant and reduces the chances of the model generating information that is not present in the video.

---

## Features

- YouTube video URL input
- Automatic YouTube video ID extraction
- Transcript extraction
- Transcript preprocessing
- Intelligent text chunking
- Mistral AI embeddings
- FAISS vector database
- Similarity-based document retrieval
- Context-aware answer generation
- Interactive Streamlit interface
- Chat-style question answering
- Retrieved context inspection
- Error handling for unavailable transcripts
- Secure API key management using environment variables

---

# Architecture

```text
                YouTube Video URL
                       |
                       v
               Extract Video ID
                       |
                       v
              Fetch Video Transcript
                       |
                       v
              Convert to Plain Text
                       |
                       v
             Recursive Text Splitter
                       |
                       v
              Create Text Chunks
                       |
                       v
              Mistral Embeddings
                       |
                       v
                 FAISS Vector Store
                       |
                       v
                Similarity Retriever
                       |
                       v
                 Relevant Chunks
                       |
                       v
                 Context + Query
                       |
                       v
               Mistral LLM
                       |
                       v
                  Final Answer
