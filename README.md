# ArVixRAG

A Q&A system over AI/ML research papers from ArXiv. You ask natural language questions, it answers grounded in actual papers with citations. Simple premise — the sophistication is entirely under the hood in how RAG is implemented, measured, and improved.

## How to set up

1. Clone the repo and navigate to the directory:
   ```bash
   git clone https://github.com/Shyamal-Shah/ArVixRAG.git
   cd ArVixRAG
   ```
2. Create a virtual environment and activate it:
   ```bash
   uv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   uv sync
   ```
4. Set up environment variables:
   ```
   cp .env.example .env
   ```
