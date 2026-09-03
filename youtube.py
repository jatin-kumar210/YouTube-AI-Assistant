from dotenv import load_dotenv

load_dotenv()

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    IpBlocked,
    RequestBlocked,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import PromptTemplate


# ==========================================================
# STEP 1: VIDEO ID
# ==========================================================

video_id = "yCA5Wzdkfag"


print("=" * 60)
print("YOUTUBE AI ASSISTANT - RAG PIPELINE")
print("=" * 60)


# ==========================================================
# STEP 2: FETCH TRANSCRIPT
# ==========================================================

print("\n[1/7] Fetching YouTube transcript...")

try:
    youtube_api = YouTubeTranscriptApi()

    # Let the API select the best available transcript.
    # Do not restrict the language.
    transcript_data = youtube_api.fetch(video_id)

    # Convert transcript to plain text
    transcript = " ".join(
        chunk.text
        for chunk in transcript_data
    )

    if not transcript.strip():
        print("Transcript is empty.")
        exit()

    print("Transcript fetched successfully.")
    print("Transcript length:", len(transcript), "characters")


except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    print("Please try another video.")
    exit()


except NoTranscriptFound:
    print("No transcript was found for this video.")
    print("Please try another video with captions.")
    exit()


except IpBlocked:
    print("YouTube has blocked your IP address.")
    print("Try another network or mobile hotspot.")
    exit()


except RequestBlocked:
    print("YouTube blocked the transcript request.")
    print("Try another network or try again later.")
    exit()


except Exception as e:
    print("Unexpected error while fetching transcript.")
    print("Error type:", type(e).__name__)
    print("Error:", e)
    exit()


# ==========================================================
# STEP 3: TEXT SPLITTING
# ==========================================================

print("\n[2/7] Splitting transcript into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = splitter.create_documents(
    [transcript]
)

print("Text splitting completed.")
print("Number of chunks:", len(chunks))


# ==========================================================
# STEP 4: MISTRAL EMBEDDINGS
# ==========================================================

print("\n[3/7] Creating Mistral embeddings...")

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

print("Mistral embedding model initialized.")


# ==========================================================
# STEP 5: FAISS VECTOR STORE
# ==========================================================

print("\n[4/7] Creating FAISS vector store...")

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created successfully.")


# ==========================================================
# STEP 6: RETRIEVER
# ==========================================================

print("\n[5/7] Creating similarity retriever...")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

print("Retriever created successfully.")
print("Search type: Similarity")
print("Documents retrieved per question: 5")


# ==========================================================
# STEP 7: MISTRAL LLM
# ==========================================================

print("\n[6/7] Initializing Mistral LLM...")

llm = ChatMistralAI(
    model="mistral-small-latest"
)

print("Mistral LLM initialized successfully.")


# ==========================================================
# PROMPT TEMPLATE
# ==========================================================

prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions about a YouTube video.

Use ONLY the provided context to answer the question.

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Instructions:
- Answer clearly and concisely.
- Use information only from the provided context.
- Do not make up information.
- Do not assume information that is not present.
- If the context does not contain enough information, say:

"I don't know based on the provided context."

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)


# ==========================================================
# SYSTEM READY
# ==========================================================

print("\n[7/7] RAG system ready.")

print("=" * 60)
print("RAG SYSTEM READY")
print("=" * 60)

print("\nAsk questions about the video.")
print("Type 'exit' to quit.")


# ==========================================================
# QUESTION-ANSWER LOOP
# ==========================================================

while True:

    question = input(
        "\nAsk a question about the video: "
    )

    # Exit
    if question.lower().strip() == "exit":
        print("Chat ended.")
        break

    # Empty question
    if not question.strip():
        print("Please enter a question.")
        continue


    # ======================================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ======================================================

    print("\nSearching relevant transcript sections...")

    try:
        results = retriever.invoke(question)

    except Exception as e:
        print("Error during retrieval.")
        print("Error type:", type(e).__name__)
        print("Error:", e)
        continue


    print(
        "Retrieved",
        len(results),
        "relevant documents."
    )


    # ======================================================
    # DISPLAY RETRIEVED DOCUMENTS
    # ======================================================

    print("\n" + "-" * 60)
    print("RETRIEVED DOCUMENTS")
    print("-" * 60)

    for i, doc in enumerate(results):

        print(
            f"\n--- Result {i + 1} ---"
        )

        print(
            doc.page_content
        )


    # ======================================================
    # COMBINE DOCUMENTS
    # ======================================================

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )


    # ======================================================
    # CREATE FINAL PROMPT
    # ======================================================

    final_prompt = prompt.format(
        context=context,
        question=question
    )


    # ======================================================
    # GENERATE ANSWER
    # ======================================================

    print("\nGenerating answer...")

    try:

        answer = llm.invoke(
            final_prompt
        )

    except Exception as e:

        print("Error while generating answer.")
        print("Error type:", type(e).__name__)
        print("Error:", e)

        continue


    # ======================================================
    # DISPLAY FINAL ANSWER
    # ======================================================

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer.content)

    print("=" * 60)``