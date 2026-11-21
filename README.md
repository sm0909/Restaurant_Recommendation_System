# Connoisseur Companion — Agentic Multimodal RAG Restaurant Recommender

A conversational assistant that recommends California restaurants from natural-language questions.
It turns unstructured restaurant write-ups and food photos into a structured knowledge base, retrieves
across text and images, reasons with a team of specialised agents, and serves everything through a
**Model Context Protocol (MCP)** application in which the agent discovers and calls its own tools at
runtime. A built-in evaluation harness measures every stage so design choices are backed by numbers.

```
unstructured text & images
        │
        ▼
[1] LLM structuring ──► schema-validated JSON knowledge base + image captions
        │
        ▼
[2] multimodal index ──► MiniLM text embeddings + CLIP image embeddings in Chroma,
        │                 metadata filtering + weighted late fusion
        ▼
[3] multi-agent workflow ──► 6 specialised agents orchestrated as a graph (LangGraph)
        │
        ▼
[4] MCP application ──► tool server + client + host (ReAct agent loop + Gradio chat UI)
        │
        ▼
[5] evaluation ──► retrieval, faithfulness, hallucination, tool-call, agent and cost metrics
```

## What I built

* **LLM data structuring with self-repair** (`evaluation/build_knowledge_base.py`): one-shot prompting
  turns each restaurant paragraph into JSON, a Pydantic schema validates it, and an LLM "repair" pass
  fixes anything that fails validation. Every extracted field is then checked against the source text,
  so structuring errors are measured instead of hidden. The run is resumable.
* **Multimodal retrieval with late fusion**: separate text (MiniLM, 384-d) and image (CLIP) indexes,
  per-modality score normalisation and weighted fusion into one ranked list.
* **Multi-agent recommendation graph**: profile → retrieval → three parallel expert analyses →
  synthesis, with shared state and graceful error handling.
* **MCP application**: a FastMCP tool server (`get_restaurant_info`, `recommend_by_vibe`, `get_review`
  plus a `culinary-map://california` resource), a client, and a host whose ReAct loop discovers tools
  with `list_tools()`, so adding a tool to the server makes it available to the agent with no host changes.
* **Provider-agnostic LLM layer** (`evaluation/llm_backend.py`): runs fully offline on a laptop CPU
  with a quantised Qwen2.5-1.5B-Instruct (llama.cpp), and records token counts and latency for every call.
* **Evaluation harness** (`evaluation/evaluate_restaurant_system.py`, test sets in
  `evaluation/eval_datasets.py`):
  * retrieval quality: BM25 vs dense vs hybrid (Hit@k, Precision@k, Recall@k, MRR, nDCG@k)
  * RAG faithfulness: an NLI model checks every answer sentence against the retrieved passages
  * hallucination reduction: closed-book LLM vs plain RAG vs grounded RAG, including "I don't know"
    behaviour on unanswerable questions
  * MCP tool-call accuracy: right tool, right arguments, malformed or unnecessary calls
  * end-to-end agent success rate
  * latency and token cost of a baseline vs an optimised configuration
* **Reliability fixes**: the MCP client and host now point at the actual server module, and a server
  smoke test (`test_mcp_server.py`) was added.

## Repository layout

| File | Stage | Purpose |
|------|-------|---------|
| `structure_text_and_multimodal_data_with_llms.py` | 1 | Prompt design and schema for structuring restaurant text (exploratory; the complete pipeline is `evaluation/build_knowledge_base.py`) |
| `process_multimodal_data_with_llms.py` | 1 | Caption recipe and review images with a vision LLM; writes `augmented_food_recipe.json` / `augmented_user_review.json` |
| `restaurant_data_management.py` | 1 | Command-line tool to view, add, edit and delete records in `structured_restaurant_data.json` |
| `multimodal_vector_index.py` | 2 | Build the text and image indexes in a persistent Chroma DB (`~/chroma_multimodal`) |
| `similarity_retrieval-1.py` | 2 | Similarity search with metadata filtering (text and image queries) |
| `multimodal_fusion_results.py` | 2 | Late fusion of text and image results into one ranking |
| `design_specialized_agents.py` | 3 | Role, goal and backstory for the six agents |
| `implement_multi_agent_systems.py` | 3 | The hybrid sequential + parallel agent graph |
| `build_chatbot_interface.py` | 3 | Chat interface with intent classification and preference extraction |
| `configure_tools_data_mcp_server.py` | 4 | MCP server (tools + resource) |
| `build_test_mcp_client.py` | 4 | MCP client that connects to the server and calls each tool |
| `design_llm_mcp_host.py` | 4 | MCP host: ReAct agent loop + Gradio chat UI |
| `test_mcp_server.py` | 4 | Server smoke test over stdio |
| `evaluation/` | 5 | Knowledge-base builder, LLM backend, test sets, evaluation harness, runner |

## Prerequisites

* Linux, macOS or WSL (Ubuntu) with **Python 3.10+**; about 4 GB of free RAM; internet access on the
  first run (to download the data, the GGUF model and the embedding/NLI models — cached afterwards).
* Optional API keys, only for the stage scripts that call hosted LLMs (see *Configuration*). The
  `evaluation/` pipeline and the MCP server need **no keys**.

## Setup

```bash
git clone <this-repo> && cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate      # optional; without it pip installs with --user

# evaluation pipeline + MCP server (local, no API keys)
pip install -r evaluation/requirements.txt

# extra packages for the stage 1-4 scripts that use hosted LLMs, Chroma and the chat UI
pip install langchain langchain-core langchain-community langchain-chroma langchain-openai langchain-ibm \
            ibm-watsonx-ai openai anthropic gradio pillow tenacity
```

## How to run

### A. Quickest path: build the knowledge base and evaluate everything (no API keys)

```bash
evaluation/run_evaluation.sh
```

The script installs the requirements, builds the knowledge base if it is missing, then runs the full
evaluation. The individual steps are:

```bash
# 1. Structure the restaurant text into structured-restaurant-data.json and link the user reviews
#    (augmented-user-review.json). Resumable: re-running continues where it stopped.
python3 evaluation/build_knowledge_base.py
python3 evaluation/build_knowledge_base.py --limit 20        # quick smoke test on 20 restaurants

# 2. Check the MCP server answers tool calls
python3 test_mcp_server.py

# 3. Run the evaluation harness
python3 evaluation/evaluate_restaurant_system.py
```

On a CPU, step 1 takes about 20–35 minutes and step 3 about 30–45 minutes.

### B. Full application with hosted LLMs

```bash
# Stage 1 - image captions for recipes and reviews (watsonx vision model)
python3 process_multimodal_data_with_llms.py
python3 restaurant_data_management.py                    # optional: curate records interactively

# Stage 2 - multimodal index, filtered retrieval, fusion
python3 multimodal_vector_index.py                       # needs structured_restaurant_data.json + augmented_food_recipe.json
python3 similarity_retrieval-1.py
python3 multimodal_fusion_results.py

# Stage 3 - multi-agent workflow and chat interface (OpenAI)
python3 implement_multi_agent_systems.py
python3 build_chatbot_interface.py

# Stage 4 - MCP client and the full chat application (watsonx + Gradio)
python3 build_test_mcp_client.py
python3 design_llm_mcp_host.py                           # opens the Gradio chat UI
```

The MCP server reads `California-Culinary-Map.txt`, `structured-restaurant-data.json` and
`augmented-user-review.json` from the repository root; `evaluation/build_knowledge_base.py` creates
all three.

## Configuration

| Variable | Used by | Meaning |
|----------|---------|---------|
| `LOCAL_GGUF_REPO`, `LOCAL_GGUF_FILE`, `LOCAL_TOKENIZER` | `evaluation/` | Swap the local model (default: Qwen2.5-1.5B-Instruct, 4-bit GGUF) |
| `LLM_THREADS` | `evaluation/` | CPU threads for llama.cpp (default: half the logical CPUs) |
| `TOKEN_PRICE_USD_PER_M` | `evaluation/` | Price per million tokens used for the cost estimate (default 0.50) |
| `NLI_MODEL`, `EMBED_MODEL` | `evaluation/` | Faithfulness judge and embedding model |
| `LLM_MODEL_ID`, `LLM_PROJECT_ID`, `LLM_API_URL` | stage 1 scripts | watsonx model, project and endpoint |
| `CALIFORNIA_CULINARY_MAP_URL`, `DATASET_URL` | `structure_text_and_multimodal_data_with_llms.py` | Source text location |
| `RECIPES_URL`, `SYNTHETIC_USER_REVIEWS_URL`, `SYNTHETIC_RECIPE_IMAGES_URL` | `process_multimodal_data_with_llms.py` | Recipe, review and image data |
| `ARCHIVE_URL` | `multimodal_vector_index.py` | Image archive for the index |
| `OPENAI_API_KEY` | stage 3 scripts | OpenAI access |
| `ANTHROPIC_API_KEY` | `build_test_mcp_client.py` | Anthropic access |
| `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` | `design_llm_mcp_host.py` | watsonx access for the chat app |

## Outputs

| Path | Contents |
|------|----------|
| `structured-restaurant-data.json`, `augmented-user-review.json`, `California-Culinary-Map.txt` | Knowledge base used by the MCP server |
| `evaluation/results/RESULTS.md` | Human-readable summary of every metric |
| `evaluation/results/metrics.json` | All metrics, plus per-question and per-task details |
| `evaluation/results/kb_structuring_metrics.json` | Structuring validity, accuracy and grounding |
| `evaluation/results/answers.jsonl` | Every RAG answer with its claim-level checks |
| `evaluation/results/*.png` | Retrieval, hallucination and latency/token charts |
| `evaluation/data/` | Resume cache and logs (not committed) |

## Design notes

* **Protocol over hardcoding.** The host treats the MCP server as a dynamic source of capabilities
  discovered at runtime, not a fixed list of calls.
* **Late fusion.** Normalising per modality before combining keeps each embedding space's own scale
  while still producing one comparable ranking.
* **Validate, then repair, then verify.** Schema validation and an LLM repair pass turn unreliable
  free-text output into dependable JSON; checking each field against the source quantifies what remains.
* **Parallelise only what is independent.** The three analysis agents share no inputs, so they run
  concurrently; profile generation and final synthesis stay sequential.
* **Measure before optimising.** Prompt size, context depth and tool-output format are tuned against
  token, latency and accuracy measurements from the evaluation harness.

## Troubleshooting

* *Slow LLM calls*: llama.cpp is sensitive to CPU contention; set `LLM_THREADS` to the number of
  free physical cores and avoid running other heavy jobs alongside.
* *`Knowledge base missing`*: run `python3 evaluation/build_knowledge_base.py` first.
* *Stage 1–4 scripts fail with authentication errors*: set the API keys listed under *Configuration*.

## Data credits

The California Culinary Map text, the synthetic user reviews and the recipe images are public sample
datasets hosted by IBM Skills Network on IBM Cloud Object Storage.
