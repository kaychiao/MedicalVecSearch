#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import dotenv

# Import local modules
from src.utils.logging_config import setup_logger
from src.vector.embeddings import VectorEmbedder
from src.vector.milvus_client import MilvusConfig
from pymilvus import MilvusClient

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

logger = setup_logger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Query NCCN guidelines from Milvus vector database')
    
    # Query parameters
    parser.add_argument('--query', '-q', type=str, required=True,
                        help='Query text to search for')
    parser.add_argument('--cancer-name', type=str, default=None,
                        help='Filter by cancer name')
    parser.add_argument('--year', type=str, default=None,
                        help='Filter by year')
    parser.add_argument('--version', type=str, default=None,
                        help='Filter by version')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of results to return (default: 5)')
    
    # Embedding model
    parser.add_argument('--embedding-model', type=str, default="all-MiniLM-L6-v2",
                        help='Embedding model to use (default: all-MiniLM-L6-v2)')
    
    # Milvus connection parameters
    parser.add_argument('--milvus-host', type=str, default=os.getenv('MILVUS_HOST', 'localhost'),
                        help='Milvus server host (default: from MILVUS_HOST env var or localhost)')
    parser.add_argument('--milvus-port', type=str, default=os.getenv('MILVUS_PORT', '19530'),
                        help='Milvus server port (default: from MILVUS_PORT env var or 19530)')
    parser.add_argument('--collection-name', type=str, default=os.getenv('MILVUS_COLLECTION', 'nccn_guidelines'),
                        help='Milvus collection name (default: from MILVUS_COLLECTION env var or nccn_guidelines)')
    
    # Output format
    parser.add_argument('--output-format', type=str, choices=['text', 'json'], default='text',
                        help='Output format (default: text)')
    
    args = parser.parse_args()
    
    return args

def format_result_as_text(result):
    """Format a search result as text."""
    output = []
    output.append(f"Score: {result['score']:.4f}")
    output.append(f"Cancer: {result.get('cancer_name', 'N/A')}")
    
    if result.get('year'):
        output.append(f"Year: {result['year']}")
    
    if result.get('version'):
        output.append(f"Version: {result['version']}")
    
    if result.get('section'):
        output.append(f"Section: {result['section']}")
    
    if result.get('source_file'):
        output.append(f"Source: {result['source_file']}")
    
    output.append("\nText:")
    output.append(result.get('text', ''))
    
    return "\n".join(output)

def search_milvus(args):
    """Search Milvus with the given arguments."""
    # Initialize vector embedder
    logger.info(f"Initializing vector embedder with model: {args.embedding_model}")
    embedder = VectorEmbedder(model_name=args.embedding_model)
    
    # Create Milvus config
    milvus_config = MilvusConfig(
        host=args.milvus_host,
        port=args.milvus_port,
        collection_name=args.collection_name
    )
    
    # Search Milvus
    filter_expr = None
    if args.cancer_name:
        filter_expr = f"cancer_name == '{args.cancer_name}'"
    if args.year:
        year_expr = f"year == '{args.year}'"
        filter_expr = year_expr if filter_expr is None else f"{filter_expr} && {year_expr}"
    if args.version:
        version_expr = f"version == '{args.version}'"
        filter_expr = version_expr if filter_expr is None else f"{filter_expr} && {version_expr}"
    
    results = search_milvus_with_config(args.query, milvus_config, embedder, args.top_k, filter_expr)
    
    return results

def search_milvus_with_config(query_text, milvus_config, embedder, top_k=5, filter_expr=None):
    """Search for vectors in Milvus that are most similar to the query text.
    
    Args:
        query_text: Query text
        milvus_config: Milvus configuration
        embedder: Vector embedder
        top_k: Number of results to return
        filter_expr: Filter expression
        
    Returns:
        list: List of search results
    """
    logger.info(f"Searching for: {query_text}")
    
    # Initialize Milvus client
    client = MilvusClient(
        uri=f"http://{milvus_config.host}:{milvus_config.port}",
        token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
    )
    
    # Check if collection exists
    if not client.has_collection(milvus_config.collection_name):
        logger.error(f"Collection {milvus_config.collection_name} does not exist")
        return []
    
    # Generate query vector
    query_embedding = embedder.embed_text(query_text)
    
    # Set search parameters
    search_params = {
        "metric_type": milvus_config.dense_metric_type,
        "params": {"nprobe": 10}
    }
    
    # Set filter expression
    expr = filter_expr if filter_expr else None
    
    # Set output fields
    output_fields = ["text", "chunk_id", "cancer_name", "year", "version", "section", "page_num", "source_file"]
    
    # Execute search
    try:
        results = client.search(
            collection_name=milvus_config.collection_name,
            data=[query_embedding],
            anns_field="dense_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields
        )
        
        # Process search results
        search_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "score": hit.score,
                }
                # Add metadata fields
                for field in output_fields:
                    if hasattr(hit, field):
                        result[field] = getattr(hit, field)
                
                search_results.append(result)
        
        return search_results
    
    except Exception as e:
        logger.error(f"Error searching Milvus: {e}")
        return []

def main():
    """Main function to query Milvus."""
    args = parse_arguments()
    
    try:
        # Search Milvus
        results = search_milvus(args)
        
        # Output results
        if not results:
            print("No results found.")
            return 0
        
        if args.output_format == 'json':
            print(json.dumps(results, indent=2))
        else:
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(format_result_as_text(result))
                print("-" * 40)
        
        return 0
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
