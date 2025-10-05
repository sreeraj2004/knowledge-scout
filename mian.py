from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile, os

# ---------- INITIAL SETUP ----------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ Missing GROQ_API_KEY in .env")

app = FastAPI(title="KnowledgeScout RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, specify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- GLOBALS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vector_store = None
doc_summary = None
llm = ChatGroq(model="gemma2-9b-it", temperature=0.2)

# ---------- ROUTES ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """Upload a PDF, process it, store embeddings, and generate a summary."""
    global vector_store, doc_summary

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    # Load and split
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    # Generate document summary
    full_text = " ".join([doc.page_content for doc in docs])
    summary_prompt = f"Summarize the following document clearly and concisely:\n\n{full_text[:12000]}"
    doc_summary = llm.invoke(summary_prompt).content

    return {
        "message": "✅ PDF uploaded, indexed, and summarized successfully!",
        "chunks": len(chunks),
        "summary_preview": doc_summary[:300] + "..."  # first 300 chars
    }

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    """Ask a question against the uploaded PDF or request the summary explicitly."""
    global vector_store, doc_summary
    if vector_store is None:
        return {"error": "No document uploaded yet!"}

    query_lower = query.lower().strip()

    # --- Explicit summary request ---
    summary_triggers = ["summary", "give me the summary", "document overview", "overview"]
    if query_lower in summary_triggers:
        return {"answer": f"📄 Document Summary:\n\n{doc_summary}"}

    # --- Retrieval-based QA for all other queries ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Flexible QA prompt
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="""
You are a helpful assistant. Use the provided context to answer the question as best as possible.
If the context is insufficient, answer with your best understanding of the document content.

Context:
{context}

Question:
{query}

Answer:
"""
    )

    # Include top chunks + document summary as context
    context_chunks = [doc.page_content[:500] for doc in vector_store.similarity_search(query, k=3)]
    context_chunks.append(doc_summary[:500])
    context = " ".join(context_chunks)

    result = llm.invoke(prompt.format(query=query, context=context))

    return {"query": query, "answer": result.content}

@app.get("/summary")
def get_summary():
    """Return the full document summary."""
    global doc_summary
    if not doc_summary:
        return {"error": "No document summary available."}
    return {"summary": doc_summary}

@app.get("/")
def root():
    return {"message": "KnowledgeScout RAG API is running!"}
