#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from pymilvus import MilvusClient

from src.utils.logging_config import setup_logger
from src.vector.milvus_client import MilvusConfig
from src.vector.milvus_operations import create_milvus_collection, store_embeddings_in_milvus
from src.data.embedding_io import load_embeddings_from_jsonl
from src.config.defaults import AppConfig

# Set up logger
logger = setup_logger(__name__)

def process_embeddings_files_in_batches(config: AppConfig) -> int:
    """Process embedding files one by one and store them in Milvus.
    
    This function processes each embedding file individually to avoid loading
    all embeddings into memory at once, which could cause memory overflow.
    
    Args:
        config: Application configuration
        
    Returns:
        int: Total number of records inserted
    """
    if not config.embeddings_dir or not os.path.exists(config.embeddings_dir):
        logger.error(f"Embeddings directory not found: {config.embeddings_dir}")
        return 0
    
    logger.info(f"Processing embedding files from directory: {config.embeddings_dir}")
    
    # Look for all embedding files in the embeddings directory
    embedding_files = list(Path(config.embeddings_dir).glob("*_embeddings.jsonl"))
    if not embedding_files:
        # Also try .json extension
        embedding_files = list(Path(config.embeddings_dir).glob("*_embeddings.json"))
    
    if not embedding_files:
        logger.error(f"No embedding files found in {config.embeddings_dir}")
        return 0
    
    logger.info(f"Found {len(embedding_files)} embedding files to process")
    
    # Initialize Milvus config
    milvus_config = MilvusConfig(
        host=config.milvus_host,
        port=config.milvus_port,
        user=config.milvus_user,
        password=config.milvus_password,
        database=config.milvus_database,
        collection_name=config.collection_name,
        vector_type=config.embedding_type,
        sparse_vector_dim=config.sparse_vector_dim if config.embedding_type == 'sparse' else None
    )
    
    # Check if we need to reset the collection
    if config.reset_collection:
        # Initialize Milvus client to drop collection if needed
        client = MilvusClient(
            uri=f"http://{milvus_config.host}:{milvus_config.port}",
            token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
        )
        
        # Select database if specified
        if hasattr(milvus_config, 'database') and milvus_config.database:
            client.use_database(db_name=milvus_config.database)
        
        # Drop collection if it exists and reset_collection is True
        if client.has_collection(milvus_config.collection_name):
            logger.info(f"Dropping collection {milvus_config.collection_name} as reset_collection is True")
            client.drop_collection(milvus_config.collection_name)
    
    # Create collection
    create_milvus_collection(milvus_config)
    
    # Process each file individually
    total_inserted = 0
    for file_path in embedding_files:
        logger.info(f"Processing file: {file_path}")
        
        # Load embeddings from the file
        try:
            embeddings, metadata, vector_dim = load_embeddings_from_jsonl(file_path)
            
            if not embeddings or not metadata:
                logger.warning(f"No valid embeddings found in {file_path}, skipping")
                continue
            
            # Store embeddings in Milvus
            inserted = store_embeddings_in_milvus(
                metadata=metadata,
                embeddings=embeddings,
                milvus_config=milvus_config,
                batch_size=config.batch_size,
                reset_collection=False  # We already handled reset_collection above
            )
            
            total_inserted += inserted
            logger.info(f"Inserted {inserted} records from {file_path}")
            
            # Clear memory
            del embeddings
            del metadata
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    logger.info(f"Total records inserted across all files: {total_inserted}")
    return total_inserted

def process_milvus_step(config: AppConfig, embeddings, metadata) -> int:
    """Process the Milvus storage step.
    
    Args:
        config: Application configuration
        embeddings: Embeddings data
        metadata: Metadata
        
    Returns:
        int: Number of records inserted
    """
    logger.info(f"Storing embeddings in Milvus collection: {config.collection_name}")
    
    # Initialize Milvus config
    milvus_config = MilvusConfig(
        host=config.milvus_host,
        port=config.milvus_port,
        user=config.milvus_user,
        password=config.milvus_password,
        database=config.milvus_database,
        collection_name=config.collection_name,
        vector_type=config.embedding_type,
        sparse_vector_dim=config.sparse_vector_dim if config.embedding_type == 'sparse' else None
    )
    
    # Check if we need to reset the collection
    if config.reset_collection:
        # Initialize Milvus client to drop collection if needed
        client = MilvusClient(
            uri=f"http://{milvus_config.host}:{milvus_config.port}",
            token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
        )
        
        # Select database if specified
        if hasattr(milvus_config, 'database') and milvus_config.database:
            client.use_database(db_name=milvus_config.database)
        
        # Drop collection if it exists and reset_collection is True
        if client.has_collection(milvus_config.collection_name):
            logger.info(f"Dropping collection {milvus_config.collection_name} as reset_collection is True")
            client.drop_collection(milvus_config.collection_name)
    
    # Create collection
    create_milvus_collection(milvus_config)
    
    # Store embeddings in Milvus
    total_inserted = store_embeddings_in_milvus(
        metadata=metadata,
        embeddings=embeddings['sparse'] if isinstance(embeddings, dict) else embeddings,
        milvus_config=milvus_config,
        batch_size=config.batch_size,
        reset_collection=False  # We already handled reset_collection above
    )
    
    logger.info(f"Stored {total_inserted} records in Milvus collection: {config.collection_name}")
    
    return total_inserted
