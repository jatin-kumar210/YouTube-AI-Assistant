from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# YoutubeTranscriptApi → used to fetch transcripts/subtitles from YouTube videos.
# TranscriptsDisabled → an exception/error class that you can catch when a video doesn't have transcripts available.


# --------------------------------------------------
# STEP 1: INDEXING
# --------------------------------------------------

video_id = "yCA5Wzdkfag"

try:

    # if you dont care which language, this returns the best one.
    transcript_data = YouTubeTranscriptApi().fetch(video_id)

    # flatten it to plain text
    transcript = " ".join(
        chunk.text for chunk in transcript_data
    )

    print("Transcript fetched successfully!")
    print("Transcript length:", len(transcript))


except TranscriptsDisabled:
    print("No caption available for this video")
    exit()

except Exception as e:
    print("Error while fetching transcript:")
    print(type(e).__name__)
    print(e)
    exit()


# --------------------------------------------------
# STEP 2: SPLIT
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = splitter.create_documents([transcript])

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# STEP 3: EMBEDDING + VECTOR STORE
# --------------------------------------------------

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector store created successfully!")


# --------------------------------------------------
# STEP 4: RETRIEVER
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# --------------------------------------------------
# STEP 5: MISTRAL LLM
# --------------------------------------------------

llm = ChatMistralAI(
    model="mistral-small-latest"
)


# --------------------------------------------------
# STEP 6: PROMPT
# --------------------------------------------------

prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions about a YouTube video.

Use the provided context to answer the question.

Context:
{context}

Question:
{question}

Instructions:
- Answer clearly and concisely.
- Use information from the context.
- Do not make up information.
- If the context does not contain enough information, say:
  "I don't know based on the provided context."

Answer:
""",
    input_variables=["context", "question"]
)


# --------------------------------------------------
# STEP 7: ASK QUESTIONS
# --------------------------------------------------

while True:

    question = input("\nAsk a question about the video (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Chat ended.")
        break


    # --------------------------------------------------
    # STEP 8: RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------

    results = retriever.invoke(question)


    # --------------------------------------------------
    # STEP 9: PRINT RETRIEVED DOCUMENTS
    # --------------------------------------------------

    print("\n------- Retrieved Documents -------")

    for i, doc in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print(doc.page_content)


    # --------------------------------------------------
    # STEP 10: COMBINE DOCUMENTS
    # --------------------------------------------------

    context = "\n\n".join(
        doc.page_content for doc in results
    )


    # --------------------------------------------------
    # STEP 11: CREATE FINAL PROMPT
    # --------------------------------------------------

    final_prompt = prompt.format(
        context=context,
        question=question
    )


    # --------------------------------------------------
    # STEP 12: GENERATE FINAL ANSWER
    # --------------------------------------------------

    answer = llm.invoke(final_prompt)


    # --------------------------------------------------
    # STEP 13: PRINT FINAL ANSWER
    # --------------------------------------------------

    print("\n------- FINAL ANSWER -------")
    print(answer.content)