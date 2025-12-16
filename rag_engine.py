import os
import json

DB_DIR = "./chroma_db"
MEMORY_FILE = "./memory/solution_history.json"
KB_PATH = "./knowledge_base/math_formulas.txt"


def get_embeddings():
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def init_vector_store():
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_core.documents import Document

    embedding_function = get_embeddings()
    documents = []

    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r") as f:
            text = f.read()
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = splitter.split_text(text)
            for t in texts:
                documents.append(
                    Document(
                        page_content=t, metadata={"source": "textbook", "type": "rule"}
                    )
                )

    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                history = json.load(f)
                for entry in history:
                    content = (
                        f"SIMILAR SOLVED PROBLEM:\n"
                        f"Q: {entry['parsed_question']}\n"
                        f"Topic: {entry['topic']}\n"
                        f"Verified Solution: {entry['final_answer']}\n"
                        f"Verifier Note: {entry['verifier_outcome']}"
                    )

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": "memory",
                                "type": "solved_example",
                                "feedback": entry.get("user_feedback", "positive"),
                            },
                        )
                    )
        except Exception as e:
            print(f"Memory Load Error: {e}")

    if documents:
        return Chroma.from_documents(
            documents, embedding_function, persist_directory=DB_DIR
        )
    else:
        return Chroma(embedding_function=embedding_function, persist_directory=DB_DIR)


def retrieve_context(query, k=3):
    from langchain_community.vectorstores import Chroma

    embedding_function = get_embeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)

    docs = db.similarity_search(query, k=k + 2)
    return [d.page_content for d in docs]


def save_full_memory_trace(data_packet):
    """
    Saves the complete lifecycle of the problem as required by the assignment.
    """
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

    history = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []

    history.append(data_packet)

    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
