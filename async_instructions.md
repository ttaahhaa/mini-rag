# Async Migration Summary

## 1. Summary: How We Moved from Sync to Async

The migration involved three core layers: **Entry Point (Routes)**, **Logic (Controllers)**, and **Infrastructure (Providers)**.

### Lifespan Management
We moved from basic initialization to an `asynccontextmanager` in `main.py`. This allows the app to await database connections (MongoDB and Vector DB) during startup, ensuring the server is fully ready before accepting traffic.

### Awaiting I/O Operations
We converted standard functions into `async def`. Instead of the server waiting idly while a database or API responds, the Event Loop is freed to handle other incoming requests.

### Parallel Execution with `asyncio.gather`
For bulk operations—like indexing hundreds of vectors—we replaced sequential `for` loops with `asyncio.gather`. This triggers multiple operations (such as Qdrant upsert calls) simultaneously.

### Non-Blocking Logic
We updated `NLPAsyncController` and `DataAsyncController` to ensure every step in the pipeline—from fetching chunks in MongoDB to searching in Qdrant—is properly awaited.

---

## 2. Where and Why to Use Multi-Threading

Even in an async app, some tasks are naturally blocking. In Python, “async” handles **I/O-bound** tasks (waiting for network or disk), but it does not handle **CPU-bound** tasks or synchronous libraries efficiently. This is where we use `asyncio.to_thread`.

### A. When to Use Multi-Threading (`asyncio.to_thread`)

- **CPU-Heavy Processing**: Tasks like text splitting, cleaning large datasets, or calculating local embeddings are CPU-intensive. Running them in the main event loop would freeze the API.
- **Synchronous SDKs**: If a library doesn’t support async (like standard `cohere` or `openai` Python clients), those calls block the server.
- **Blocking File I/O**: Standard `os` or `open()` calls are blocking. While we use `aiofiles` for streaming, small OS operations like `os.path.getsize` are offloaded to threads for better performance.

### B. Where It Was Applied in the App

- **Embeddings**: We wrapped `embed_batch` and `embed_text` in `to_thread`. This prevents the 300–500 ms API latency of the embedding provider from halting the rest of the server.
- **File Loading**: In `ProcessAsyncController`, reading large PDFs from disk was moved to a thread to keep ingestion smooth.

---

## 3. The “Async 101” Rules for Development

To maintain this architecture as you add new features (like reranking or agentic logic), follow these rules:

| Scenario                  | Use This Tool       | Reason                                                |
|---------------------------|---------------------|-------------------------------------------------------|
| Database / Network call   | `await`             | Releases the loop while waiting for a response.       |
| Multiple independent calls| `asyncio.gather`    | Runs them in parallel for speed.                      |
| Blocking Sync library     | `asyncio.to_thread` | Prevents the library from freezing the API.           |
| CPU-heavy math / parsing  | `asyncio.to_thread` | Keeps the Event Loop responsive for other users.      |

---

## Next Step for Your Project

Since the core pipeline is now highly efficient, the next improvement is **Async Middleware**. This can log the processing time of every request and help you pinpoint bottlenecks in your “thousands per minute” workflow.
