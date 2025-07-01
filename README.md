# MedicalVecSearch

A comprehensive system for processing medical documents (including NCCN guidelines) and creating searchable vector representations using Milvus.

## Overview

MedicalVecSearch extracts text from medical PDFs, chunks the content semantically, generates vector embeddings, and stores them in a Milvus vector database for efficient semantic search. The system also supports BM25 full-text search capabilities.

## Features

- **PDF Processing**: Extract text and structure from medical PDFs using Docling
- **Semantic Chunking**: Split documents into meaningful chunks with configurable size and overlap
- **Vector Embeddings**: Generate dense, sparse, or hybrid vector embeddings
- **Vector Database**: Store and search vectors using Milvus 2.5.x
- **BM25 Full-Text Search**: Perform keyword-based search using Milvus BM25 functionality
- **Modular Pipeline**: Run specific steps (extract, chunk, embed, store) or the entire pipeline
- **Batch Processing**: Handle large documents with memory-efficient batch processing
- **Flexible Configuration**: Configure via command line, environment variables, or config files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MedicalVecSearch.git
cd MedicalVecSearch
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables (copy from example):
```bash
cp env.example .env
# Edit .env with your configuration
```

## Usage

### Process Documents and Store in Milvus

```bash
python nccn_to_milvus.py --input /path/to/pdfs --models /path/to/docling/models --steps extract,generate_embeddings,store_in_milvus
```

### Load Existing Embeddings into Milvus

```bash
python nccn_to_milvus.py --jsonl-file /path/to/embeddings.jsonl --steps load_jsonl
```

### Run Vector Search Query

```bash
python demo_query.py
```

### Run BM25 Full-Text Search

```bash
# Create collection, load data, and search
python demo_bm25_search.py --mode all --query "your search query"

# Just search in existing collection
python demo_bm25_search.py --mode search --query "your search query"
```

## Pipeline Steps

1. **EXTRACT**: Convert PDFs to text using Docling
2. **CHUNK**: Split text into semantic chunks
3. **EMBED**: Generate vector embeddings for text chunks
4. **STORE**: Store embeddings and metadata in Milvus

## Directory Structure

```
MedicalVecSearch/
├── nccn_to_milvus.py      # Main script for processing documents
├── demo_query.py          # Vector search demo
├── demo_bm25_search.py    # BM25 full-text search demo
├── load_chunks_to_bm25.py # Load text chunks for BM25 search
├── src/                   # Source code
│   ├── config/            # Configuration handling
│   ├── data/              # Data processing utilities
│   ├── pdf/               # PDF processing modules
│   ├── pipeline/          # Pipeline workflow
│   ├── utils/             # Utility functions
│   └── vector/            # Vector operations and Milvus integration
├── tmp/                   # Temporary files and intermediate results
│   ├── text/              # Extracted text from PDFs
│   ├── markdown/          # Markdown content from PDFs
│   ├── metadata/          # JSON metadata about extraction
│   ├── chunks/            # JSON files with chunked text data
│   └── embeddings/        # JSON files with vector embeddings
└── test_docs/             # Test documents
```

## Configuration

The system can be configured through:
1. Command line arguments
2. Environment variables (in `.env` file)
3. Default configuration in `src/config/defaults.py`

Key configuration options:
- Milvus connection settings
- Embedding model selection
- Vector dimensions
- Chunking parameters
- Processing steps

## Requirements

- Python 3.8+
- Milvus 2.5.x
- See `requirements.txt` for Python package dependencies

## License

[Your License]

## Contributors

[Your Name/Organization]