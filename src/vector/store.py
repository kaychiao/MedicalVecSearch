#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time

# Import for Milvus
from pymilvus import MilvusClient, DataType

# Import local modules
from src.vector.embeddings import VectorEmbedder
from src.vector.milvus_client import MilvusConfig
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def process_and_store(
    chunks: List,
    embedder: VectorEmbedder,
    milvus_config: MilvusConfig,
    batch_size: int = 100
) -> int:
    """Process chunks, generate embeddings, and store in Milvus.
    
    Returns the total number of records inserted.
    """
    logger.info(f"Processing {len(chunks)} chunks and storing in Milvus...")
    
    # 初始化Milvus客户端
    client = MilvusClient(
        uri=f"http://{milvus_config.host}:{milvus_config.port}",
        token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
    )
    
    # 确保集合存在
    if not client.has_collection(milvus_config.collection_name):
        logger.error(f"Collection {milvus_config.collection_name} does not exist")
        return 0
    
    total_inserted = 0
    
    # Process in batches to avoid memory issues
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")
        
        # Generate embeddings and prepare metadata
        embedding_result = embedder.embed_chunks(batch)
        
        # Handle different embedding types
        if embedder.embedding_type == "hybrid":
            # For hybrid embeddings, we get a dict with 'dense' and 'sparse' keys
            # Each containing (embeddings, metadata) tuple
            dense_embeddings, dense_metadata = embedding_result['dense']
            sparse_embeddings, sparse_metadata = embedding_result['sparse']
            
            # Ensure metadata is consistent
            if dense_metadata != sparse_metadata:
                logger.warning("Metadata mismatch between dense and sparse embeddings")
            
            # 准备批量插入的数据
            batch_data = []
            for j, meta in enumerate(dense_metadata):
                entity = meta.copy()
                entity['dense_embedding'] = dense_embeddings[j]
                entity['sparse_embedding'] = sparse_embeddings[j]
                batch_data.append(entity)
        else:
            # For single type embeddings (dense or sparse)
            embeddings, metadata = embedding_result
            
            # 准备批量插入的数据
            batch_data = []
            vector_field = 'dense_embedding' if embedder.embedding_type == 'dense' else 'sparse_embedding'
            for j, meta in enumerate(metadata):
                entity = meta.copy()
                entity[vector_field] = embeddings[j]
                batch_data.append(entity)
        
        try:
            # 使用Milvus客户端插入数据
            result = client.insert(
                collection_name=milvus_config.collection_name,
                data=batch_data
            )
            inserted = result.insert_count
            total_inserted += inserted
            logger.info(f"Inserted batch {i//batch_size + 1}: {inserted} records")
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    logger.info(f"Total inserted: {total_inserted} records")
    return total_inserted
