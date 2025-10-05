from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile, os, json
from typing import Optional

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

# ---------- LAZY COMPONENTS (for serverless efficiency) ----------
class LazyComponents:
    def __init__(self):
        self._embeddings = None
        self._llm = None
        self._vector_store = None
        self._doc_summary = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGroq(model="gemma2-9b-it", temperature=0.2)
        return self._llm

    @property
    def vector_store(self):
        return self._vector_store

    @vector_store.setter
    def vector_store(self, value):
        self._vector_store = value

    @property
    def doc_summary(self):
        return self._doc_summary

    @doc_summary.setter
    def doc_summary(self, value):
        self._doc_summary = value

components = LazyComponents()

# ---------- ROUTES ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """Upload a PDF, process it, store embeddings, and generate a summary.
    Note: In serverless (Vercel), state resets per request. For persistence, integrate Vercel KV/Redis."""
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load and split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path)  # Clean up temp file immediately

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise HTTPException(status_code=400, detail="PDF is empty or unreadable.")

        # Create vector store (lazy embeddings)
        components.vector_store = FAISS.from_documents(chunks, embedding=components.embeddings)

        # Generate document summary (lazy LLM)
        full_text = " ".join([doc.page_content for doc in docs])
        summary_prompt = f"Summarize the following document clearly and concisely:\n\n{full_text[:12000]}"
        components.doc_summary = components.llm.invoke(summary_prompt).content

        return {
            "message": "✅ PDF uploaded, indexed, and summarized successfully!",
            "chunks": len(chunks),
            "summary_preview": components.doc_summary[:300] + "..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    """Ask a question against the uploaded PDF or request the summary explicitly."""
    if components.vector_store is None or components.doc_summary is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No document uploaded yet! Please upload a PDF first."}
        )

    query_lower = query.lower().strip()

    # --- Explicit summary request ---
    summary_triggers = ["summary", "give me the summary", "document overview", "overview"]
    if any(trigger in query_lower for trigger in summary_triggers):
        return {"answer": f"📄 Document Summary:\n\n{components.doc_summary}"}

    # --- Retrieval-based QA for all other queries ---
    # Removed unused RetrievalQA chain for simplicity/efficiency

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
    context_chunks = [doc.page_content[:500] for doc in components.vector_store.similarity_search(query, k=3)]
    context_chunks.append(components.doc_summary[:500])
    context = " ".join(context_chunks)

    result = components.llm.invoke(prompt.format(query=query, context=context))

    return {"query": query, "answer": result.content}

@app.get("/summary")
def get_summary():
    """Return the full document summary."""
    if not components.doc_summary:
        return JSONResponse(
            status_code=400,
            content={"error": "No document summary available."}
        )
    return {"summary": components.doc_summary}

@app.get("/")
def root():
    return {"message": "KnowledgeScout RAG API is running!"}