#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from pathlib import Path
import dotenv

# Import local modules
from src.utils.logging_config import setup_logger
from src.pipeline.workflow import run_pipeline
from src.config.defaults import load_config, AppConfig

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

logger = setup_logger(__name__)

def parse_arguments():
    """Parse command line arguments, keeping only the most commonly used parameters.
    Other parameters can be set via config file or environment variables.
    """
    config = load_config()  # Load default configuration
    
    parser = argparse.ArgumentParser(
        description='Process medical documents and store them in Milvus vector database',
        epilog='Most parameters can be set via .env file or environment variables, only specify necessary parameters on the command line.'
    )
    
    # Required input/output parameters
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Input directory containing PDF files or intermediate files')
    parser.add_argument('--models', '-m', type=str, required=False,
                        help='Path to Docling model files (optional)')
    parser.add_argument('--intermediate-dir', '-id', type=str, required=False,
                        help='Directory to save intermediate results (optional)')
    
    # Optional file paths
    parser.add_argument('--chunks-file', '-cf', type=str, required=False,
                        help='Path to save/load chunked text data (JSON format)')
    parser.add_argument('--embeddings-file', '-ef', type=str, required=False,
                        help='Path to save/load embedding data (JSON format)')
    parser.add_argument('--jsonl-file', type=str, default=None,
                        help='Path to JSONL file containing embeddings to load directly into Milvus')
    
    # Common processing options
    parser.add_argument('--steps', type=str, default=config.steps,
                        help=f'List of steps to run, comma-separated (default: {config.steps})')
    parser.add_argument('--start-from', type=str, default=config.start_from,
                        help='Start processing from this step')
    parser.add_argument('--reset-collection', action='store_true', default=config.reset_collection,
                        help='Reset (delete and recreate) Milvus collection')
    
    # Simplified parameter groups
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--no-save-intermediate', action='store_true',
                        help='Disable saving intermediate results for each file')
    advanced_group.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration (use CPU only)')
    advanced_group.add_argument('--batch-size', type=int, default=config.batch_size,
                        help=f'Batch processing size (default: {config.batch_size})')
    
    # Embedding model options
    model_group = parser.add_argument_group('Embedding Model Options')
    model_group.add_argument('--embedding-model', type=str, default=config.embedding_model,
                        help=f'Dense embedding model (default: {config.embedding_model})')
    model_group.add_argument('--embedding-type', type=str, 
                        choices=['dense', 'sparse', 'hybrid'], default=config.embedding_type,
                        help=f'Embedding type: dense, sparse, or hybrid (default: {config.embedding_type})')
    
    # Milvus connection options
    milvus_group = parser.add_argument_group('Milvus Connection Options')
    milvus_group.add_argument('--milvus-host', type=str, default=config.milvus_host,
                        help=f'Milvus server host (default: {config.milvus_host})')
    milvus_group.add_argument('--collection-name', type=str, default=config.collection_name,
                        help=f'Milvus collection name (default: {config.collection_name})')
    
    return parser.parse_args()

def main():
    """Main function to process documents and store them in Milvus."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Update configuration with command line arguments
    config.update_from_args(args)
    
    # Print configuration information
    logger.info("Running with the following configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Run pipeline
    return run_pipeline(config)

if __name__ == "__main__":
    exit(main())
