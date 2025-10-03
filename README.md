# Vectorstores

This repository collects everything related to Vectorstores, unless it is a separate project.

## Overview

Vector stores (or vector databases) are a crucial component in modern AI applications, particularly for:
- Semantic search
- Question answering systems
- Retrieval-Augmented Generation (RAG)
- Chatbots with memory
- Document analysis

## Examples

### FAISS Example

The `faiss_example.py` demonstrates how to:
- Create a vector store using FAISS (Facebook AI Similarity Search)
- Load and process PDF documents
- Split documents into chunks for embedding
- Create embeddings using OpenAI
- Build a conversational retrieval chain with memory
- Query the vector store with natural language questions

## Setup

1. Clone this repository:
```bash
git clone https://github.com/gvtsch/Vectorstores.git
cd Vectorstores
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. Run the FAISS example:
```bash
python faiss_example.py
```

## Requirements

- Python 3.8+
- OpenAI API key

## Dependencies

See `requirements.txt` for the full list of dependencies, which includes:
- LangChain and LangChain Community
- FAISS (CPU version)
- OpenAI
- PyPDF for PDF processing
- ReportLab for PDF generation
- python-dotenv for environment variable management

## Contributing

Contributions are welcome! If you have examples or implementations of other vector stores, feel free to submit a pull request.

## License

This project is open source and available under the MIT License.