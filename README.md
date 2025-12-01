## Overview

This project contains experiments with LangChain, DeepAgents, and OpenRouter to build:

- **A web research agent** that can browse the internet and answer questions with up-to-date information.
- **A RAG (Retrieval-Augmented Generation) system** that lets you chat with your own research papers (PDFs).

All of the core logic currently lives in the notebook `basic_LC.ipynb`.

---

## 1. Web Research Agent (Internet Search)

### What it does
- Uses `ddgs` (DuckDuckGo Search) as a free web search backend.
- Wraps search in a Python tool function `internet_search`.
- Creates an LLM via `ChatOpenAI` from `langchain_openai`, configured to use OpenRouter.
- Builds an agent with `deepagents.create_deep_agent` that:
  - Calls `internet_search` as needed.
  - Is guided by system instructions to always verify time-sensitive information via the web.

### Key pieces
- **Environment variables**:
  - `OPENAI_API_KEY` – OpenRouter API key.
  - `OPENAI_BASE_URL` – set to `https://openrouter.ai/api/v1`.
- **Core imports**:
  - `from langchain_openai import ChatOpenAI`
  - `from deepagents import create_deep_agent`
  - `from ddgs import DDGS`
- **Tool**:
  - `internet_search(query: str, max_results: int = 5)`  
    Uses `DDGS().text(...)` to fetch results and returns formatted snippets (title, URL, body).
- **Agent**:
  - Created with `create_deep_agent(tools=[internet_search], system_prompt=research_instructions, model=llm)`.
  - `research_instructions` strongly emphasize:
    - Training data is outdated.
    - Time-sensitive questions must be answered via web search.
    - Always cite dates and sources.

### Typical usage (in the notebook)
1. Install and import dependencies (`ddgs`, `langchain-openai`, `deepagents`).
2. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
3. Define `internet_search`.
4. Define `research_instructions`.
5. Create the LLM and agent.
6. Call:
   ```python
   result = agent.invoke({
       "messages": [{"role": "user", "content": "Some question that requires web search"}]
   })
   print(result["messages"][-1].content)
   ```

---

## 2. RAG System – Chat With Your Research Papers

### What it does
- Loads your own PDF research papers from a `pdfs/` directory.
- Splits them into overlapping text chunks.
- Creates embeddings for each chunk using OpenAI embeddings (via `langchain_openai`).
- Stores embeddings in a persistent ChromaDB vector store (`chroma_db/`).
- Exposes a `search_documents` tool that retrieves the most relevant chunks for a query.
- Builds a RAG agent that:
  - Uses `search_documents` as the primary tool.
  - Can also fall back to `internet_search` for extra context.

### Libraries used
- `pypdf` – PDF parsing through `PyPDFLoader`.
- `langchain_community.document_loaders.PyPDFLoader`
- `langchain_text_splitters.RecursiveCharacterTextSplitter`
- `langchain_openai.OpenAIEmbeddings`
- `langchain_community.vectorstores.Chroma`
- `deepagents.create_deep_agent`
- `langchain_openai.ChatOpenAI`

### Data flow
1. **Load PDFs**
   ```python
   chunks = load_pdfs_from_directory("./pdfs")
   ```
   - Looks for all `.pdf` files under `./pdfs`.
   - Loads each with `PyPDFLoader`.
   - Adds `source_file` metadata for each page.
   - Splits into chunks with `RecursiveCharacterTextSplitter` (chunk size ~1000 chars, 200 overlap).

2. **Create vector store**
   ```python
   vectorstore = create_vector_store(chunks, persist_directory="./chroma_db")
   ```
   - Uses `Chroma.from_documents`.
   - Persists to `./chroma_db` so you do not have to re-embed every run.

3. **Reload existing vector store (optional)**
   ```python
   vectorstore = Chroma(
       persist_directory="./chroma_db",
       embedding_function=embeddings
   )
   ```

4. **Search documents tool**
   ```python
   def search_documents(query: str, k: int = 5):
       # Uses vectorstore.similarity_search_with_score
       # Returns formatted text with:
       # - relevance score
       # - source file name
       # - page number
       # - content snippet
   ```

5. **RAG agent**
   ```python
   rag_llm = ChatOpenAI(
       model="openai/gpt-4o-mini",
       api_key=os.environ["OPENAI_API_KEY"],
       base_url=os.environ["OPENAI_BASE_URL"]
   )

   rag_agent = create_deep_agent(
       tools=[search_documents, internet_search],
       system_prompt=rag_instructions,
       model=rag_llm
   )
   ```

   - `rag_instructions` tell the agent to:
     - Always use `search_documents` first for questions about the PDFs.
     - Organize answers by paper title when multiple papers are involved.
     - Use clear headings, cite file names and page numbers, and quote relevant sections.

6. **Example RAG query**
   ```python
   query = """For each research paper, please provide:
   1. The paper title
   2. The main topics and findings from that specific paper

   Organize your response by separating the findings for each paper
   with clear headings using the paper titles."""

   result = rag_agent.invoke({
       "messages": [{"role": "user", "content": query}]
   })
   print(result["messages"][-1].content)
   ```

---

## 3. Project Structure

- `basic_LC.ipynb`  
  Main notebook containing:
  - Web research agent using `internet_search`.
  - RAG pipeline (PDF loading, splitting, embeddings, ChromaDB).
  - RAG agent using `search_documents` and `internet_search`.
- `asd.ipynb`  
  Earlier experimentation notebook for agents and web search.
- `pdfs/`  
  Folder where you place your research paper PDFs.
- `chroma_db/`  
  Persistent vector store generated from your PDFs.

---

## 4. Setup Instructions (Minimal)

From within the notebook (already present as cells):

1. Install dependencies:
   - `ddgs`
   - `langchain-openai`
   - `pypdf`
   - `chromadb`
   - `langchain-community`
   - `langchain-text-splitters`
   - `deepagents`

2. Set environment variables:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-openrouter-key"
   os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
   ```

3. Add PDFs to the `pdfs/` folder.

4. In `basic_LC.ipynb`, run the RAG cells in order:
   - Imports
   - Embeddings
   - `load_pdfs_from_directory`
   - `create_vector_store`
   - `search_documents`
   - `rag_instructions`
   - `rag_agent` creation
   - Test query cell

5. Use the test cells to:
   - Ask general web questions with the web research agent.
   - Ask document-specific questions with the RAG agent.

This README is a high-level summary of what the `basic_LC.ipynb` notebook currently does and how to use it.


