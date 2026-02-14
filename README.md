# GenAI Studio SDK

A Python wrapper for Purdue's GenAI Studio API — chat completions, streaming, multi-turn conversations, embeddings, RAG pipelines, and a full CLI. Built for the Open WebUI + LiteLLM + Ollama stack at `genai.rcac.purdue.edu`.

## Setup

```bash
pip install -r requirements.txt
export GENAI_STUDIO_API_KEY="your-key"
```

Get your API key from **GenAI Studio → Settings → Account → API Keys**.

## Quick Start

```python
from genai_studio import GenAIStudio

ai = GenAIStudio()
ai.select_model("gemma3:12b")

# Chat
response = ai.chat("What is a p-value?")

# Stream
for chunk in ai.chat_stream("Explain regression"):
    print(chunk, end="", flush=True)

# RAG — ground responses in your documents
file = ai.upload_file("notes.pdf")
kb = ai.create_knowledge_base("My Notes")
ai.add_file_to_knowledge_base(kb.id, file.id)

import time; time.sleep(10)  # Wait for indexing

response = ai.chat("Summarize chapter 1", collections=[kb.id])
```

## CLI

```bash
python genai_studio.py models                              # List models
python genai_studio.py chat -m gemma3:12b "What is AI?"    # Single query
python genai_studio.py chat -m gemma3:12b -i --stream      # Interactive
python genai_studio.py embed -m llama3.2:latest "a" "b" --similarity
python genai_studio.py rag upload notes.pdf                # RAG workflow
python genai_studio.py health                              # Connection check
```

## Tests

```bash
python test_full_suite.py                    # Default: 3 chat models, 1 embed model
python test_full_suite.py --all-models       # Every available model
python test_full_suite.py --skip-rag         # Fast core-only run
python test_full_suite.py -v                 # Verbose output
```

## Documentation

Full developer guide with architecture diagrams, code examples, method reference, and troubleshooting:

**[Open the Guide →](https://htmlpreview.github.io/?https://github.com/treese41528/genai-studio-sdk/blob/main/docs/guide.html)**

Or open `docs/guide.html` locally in any browser.

## What's In The Box

| File | Description |
|------|-------------|
| `genai_studio.py` | The SDK — single-file module, library + CLI |
| `test_full_suite.py` | 276-test suite across all models and file types |
| `docs/guide.html` | Interactive developer guide |
| `requirements.txt` | Dependencies: `openai`, `httpx`, `numpy` |

## License

MIT