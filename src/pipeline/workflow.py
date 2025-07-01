#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from src.utils.logging_config import setup_logger
from src.vector.milvus_client import MilvusConfig
from src.data.file_operations import load_chunks_from_file
from src.data.embedding_io import (
    save_embeddings_per_pdf, 
)
from src.vector.milvus_operations import (
    load_and_store_jsonl,
)
from src.vector.batch_processing import (
    process_embeddings_files_in_batches,
    process_milvus_step
)
from src.config.defaults import AppConfig

logger = setup_logger(__name__)

def process_jsonl_direct(jsonl_file: str, milvus_config: MilvusConfig, batch_size: int = 100) -> int:
    """Load embeddings directly from a JSONL file into the Milvus database.
    
    Args:
        jsonl_file: Path to the JSONL file
        milvus_config: Milvus configuration
        batch_size: Batch processing size
        
    Returns:
        int: Number of records inserted
    """
    logger.info(f"Loading embeddings directly from JSONL file: {jsonl_file}")
    
    # Load and store JSONL file directly
    total_inserted = load_and_store_jsonl(
        file_path=jsonl_file,
        milvus_config=milvus_config,
        batch_size=batch_size
    )
    
    logger.info(f"Completed loading JSONL. Total records inserted: {total_inserted}")
    return total_inserted

def process_extract_step(config: AppConfig) -> List:
    """Process the text extraction step.
    
    Args:
        config: Application configuration
        
    Returns:
        List: List of extracted text chunks
    """
    from src.pdf.processor import process_directory
    
    logger.info(f"Extracting text from PDFs in {config.input}")
    
    # Process directory of PDFs
    chunks = process_directory(
        input_dir=config.input,
        models_dir=config.models,
        output_dir=config.intermediate_dir,
        max_chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        batch_size=config.batch_size,
        save_intermediate=config.save_intermediate,
        use_gpu=config.use_gpu
    )
    
    logger.info(f"Extracted {len(chunks)} chunks from PDFs")
    
    # Ensure chunks directory exists
    if config.chunks_dir:
        os.makedirs(config.chunks_dir, exist_ok=True)
        # Note: process_directory already saved individual PDF chunk files
        # No need to save merged files, saving storage space
        logger.info(f"Individual chunk files were saved by process_directory")
        
    return chunks

def process_embedding_step(config: AppConfig, chunks) -> Tuple[Union[List, Dict], List, Any]:
    """Process the embedding generation step.
    
    Args:
        config: Application configuration
        chunks: List of text chunks
        
    Returns:
        Tuple: (embeddings, metadata, vector dimensions)
    """
    from src.vector.embeddings import VectorEmbedder
    
    logger.info(f"Generating {config.embedding_type} embeddings for {len(chunks)} chunks")
    
    # Initialize embedder
    embedder = VectorEmbedder(
        model_name=config.embedding_model,
        embedding_type=config.embedding_type,
        sparse_model_name=config.sparse_model,
        dense_vector_dim=config.dense_vector_dim,
        sparse_vector_dim=config.sparse_vector_dim
    )
    
    # Generate embeddings and get metadata
    embedding_result = embedder.embed_chunks(chunks)
    
    # Get vector dimension(s)
    vector_dim = embedder.vector_dim
    
    if config.embedding_type == 'hybrid':
        # For hybrid embeddings, the result is a dict with 'dense' and 'sparse' keys
        # Each containing a tuple of (embeddings, metadata)
        dense_embeddings, dense_metadata = embedding_result['dense']
        sparse_embeddings, sparse_metadata = embedding_result['sparse']
        
        # Ensure metadata is consistent
        if dense_metadata != sparse_metadata:
            logger.warning("Metadata for dense and sparse embeddings do not match. Using dense metadata.")
        
        embeddings = {
            'dense': dense_embeddings,
            'sparse': sparse_embeddings
        }
        metadata = dense_metadata
    else:
        # For dense or sparse embeddings, the result is a tuple of (embeddings, metadata)
        embeddings, metadata = embedding_result
    
    # Save embeddings to file
    if config.embeddings_dir:
        os.makedirs(config.embeddings_dir, exist_ok=True)
        
        # Save individual embeddings per PDF
        save_embeddings_per_pdf(embeddings, metadata, config.embeddings_dir, config.embedding_type)
        
        # Save merged embeddings file
        # embeddings_file = os.path.join(config.embeddings_dir, "document_embeddings.json")
        # save_embeddings_to_file(embeddings, metadata, embeddings_file, config.embedding_type)
        # logger.info(f"Saved embeddings to {embeddings_file}")
    
    return embeddings, metadata, vector_dim

def verify_metadata_consistency(intermediate_dir: str, metadata: List):
    """Verify metadata consistency.
    
    Args:
        intermediate_dir: Intermediate files directory
        metadata: List of metadata
    """
    metadata_dir = Path(intermediate_dir) / "metadata"
    if metadata_dir.exists():
        # Get all metadata files
        metadata_files = list(metadata_dir.glob("*_info.json"))
        
        if metadata_files:
            # Load the first metadata file to compare
            with open(metadata_files[0], 'r') as f:
                first_metadata = json.load(f)
            
            # Compare fields with other metadata files
            for metadata_file in metadata_files[1:]:
                with open(metadata_file, 'r') as f:
                    current_metadata = json.load(f)
                
                # Check for field consistency
                for key in first_metadata:
                    if key in current_metadata and first_metadata[key] != current_metadata[key]:
                        logger.warning(f"Metadata inconsistency: {key} differs between files")
                        logger.warning(f"  {metadata_files[0].name}: {first_metadata[key]}")
                        logger.warning(f"  {metadata_file.name}: {current_metadata[key]}")
    else:
        logger.info(f"Metadata directory not found: {metadata_dir}")

def determine_steps_to_run(config: AppConfig) -> List[str]:
    """Determine which steps to run.
    
    Args:
        config: Application configuration
        
    Returns:
        List[str]: List of steps to run
    """
    all_steps = {
        'extract': 'EXTRACT_TEXT',
        'chunk': 'CHUNK_TEXT',
        'generate_embeddings': 'GENERATE_EMBEDDINGS',
        'store_in_milvus': 'STORE_IN_MILVUS',
        'load_jsonl': 'LOAD_JSONL'
    }
    
    steps_arg = config.steps.split(',')
    start_from = config.start_from
    
    # Validate steps
    for step in steps_arg:
        if step not in all_steps:
            logger.error(f"Invalid step: {step}. Valid steps are: {', '.join(all_steps.keys())}")
            return []
    
    # Determine which steps to run based on start_from
    steps_to_run = []
    if start_from:
        started = False
        for step in all_steps:
            if step == start_from:
                started = True
            if started and step in steps_arg:
                steps_to_run.append(step)
    else:
        steps_to_run = steps_arg
    
    logger.info(f"Running steps: {', '.join(steps_to_run)}")
    return steps_to_run

def run_pipeline(config: AppConfig) -> int:
    """Run the processing pipeline.
    
    Args:
        config: Application configuration
        
    Returns:
        int: Exit code, 0 indicates success
    """
    try:
        # Determine which steps to run
        steps_to_run = determine_steps_to_run(config)
        if not steps_to_run:
            return 1
        
        # Special case for direct JSONL loading
        if 'load_jsonl' in steps_to_run and config.jsonl_file:
            # Initialize Milvus config
            milvus_config = MilvusConfig(
                host=config.milvus_host,
                port=config.milvus_port,
                user=config.milvus_user,
                password=config.milvus_password,
                database=config.milvus_database,
                collection_name=config.collection_name,
                dense_vector_dim=config.dense_vector_dim,
                sparse_vector_dim=config.sparse_vector_dim
            )
            
            process_jsonl_direct(config.jsonl_file, milvus_config, config.batch_size)
            return 0
        
        # Initialize variables
        chunks = None
        embeddings = None
        metadata = None
        vector_dim = None
        
        # Step 1: Extract text from PDFs
        if 'extract' in steps_to_run:
            chunks = process_extract_step(config)
        else:
            # Try to load chunks from directory if not extracting
            chunks = []
            if config.chunks_dir and os.path.exists(config.chunks_dir):
                # Look for all *_chunks.json files in the chunks directory
                chunk_files = list(Path(config.chunks_dir).glob("*_chunks.json"))
                if chunk_files:
                    logger.info(f"Found {len(chunk_files)} chunk files in {config.chunks_dir}")
                    for chunk_file in chunk_files:
                        logger.info(f"Loading chunks from {chunk_file}")
                        file_chunks = load_chunks_from_file(chunk_file)
                        chunks.extend(file_chunks)
                    logger.info(f"Loaded a total of {len(chunks)} chunks from all files")
            
            # Only show error if we need chunks for the next step
            if 'generate_embeddings' in steps_to_run and not chunks:
                logger.error("No chunks available and extract step not selected")
                return 1
        
        # Step 2: Generate embeddings
        if 'generate_embeddings' in steps_to_run and chunks:
            embeddings, metadata, vector_dim = process_embedding_step(config, chunks)
        
        # Step 3: Store in Milvus
        if 'store_in_milvus' in steps_to_run:
            # If we have embeddings from the previous step, store them
            if embeddings and metadata:
                process_milvus_step(config, embeddings, metadata)
            else:
                # Process embedding files in batches to avoid memory overflow
                logger.info("No embeddings from previous step, processing embedding files in batches")
                process_embeddings_files_in_batches(config)
        
        return 0
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1
