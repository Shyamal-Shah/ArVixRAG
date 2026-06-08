# ArVixRAG

ArVixRAG is a Retrieval-Augmented Generation (RAG) system designed to provide accurate and contextually relevant responses to user queries by leveraging a combination of retrieval and generation techniques. The system retrieves relevant information from a knowledge base and uses it to generate informative and coherent responses.

## Features

### Core Features

- **Retrieval-Augmented Generation**: Combines retrieval and generation techniques to provide accurate and contextually relevant responses.
- **Knowledge Base Integration (arVix)**: Integrates with a knowledge base to retrieve relevant information for generating responses.
- **Contextual Understanding**: Understands the context of user queries to provide more accurate and relevant responses.
- **Multi-turn Conversations**: Supports multi-turn conversations, allowing for more natural and engaging interactions.
- **Grounding and Attribution**: Provides grounding and attribution for generated responses, ensuring transparency and reliability.

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
