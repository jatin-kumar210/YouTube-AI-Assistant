# YouTube AI Assistant 

(live:https://readmemd-pnt3rj954vhyu8efywognn.streamlit.app/)
<p align="center">
  <strong>AI-powered question answering for YouTube videos using Retrieval-Augmented Generation (RAG)</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a>
</p>

---

## Overview

YouTube AI Assistant is a Retrieval-Augmented Generation (RAG) application
that allows users to interact with YouTube videos through natural-language
questions.

The application extracts the video's transcript, divides it into meaningful
chunks, generates vector embeddings, and stores them in a FAISS vector
database. When a user asks a question, the system retrieves the most relevant
parts of the transcript and provides them to a Mistral AI language model to
generate a contextual answer.

The application is built with Python and Streamlit and provides an interactive
interface for processing videos and asking questions.

---

## Demo

Watch the application in action:

**[▶ Watch Demo](demo.mp4)**

---

## Features

- YouTube transcript extraction
- Natural-language question answering
- Retrieval-Augmented Generation (RAG)
- Semantic similarity search with FAISS
- Mistral AI embeddings
- Mistral AI language model
- Interactive Streamlit interface
- Context-aware responses
- Conversation history
- Environment-based API key configuration

---

## Architecture

```text
                    YouTube Video
                         │
                         ▼
                 Extract Video ID
                         │
                         ▼
                  Fetch Transcript
                         │
                         ▼
                   Text Processing
                         │
                         ▼
                  Text Chunking
                         │
                         ▼
               Mistral AI Embeddings
                         │
                         ▼
                   FAISS Index
                         │
                         │
User Question ──────────┤
                         ▼
                Similarity Retrieval
                         │
                         ▼
                 Relevant Context
                         │
                         ▼
                 Mistral AI LLM
                         │
                         ▼
                   Final Answer
