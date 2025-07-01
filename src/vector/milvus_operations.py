#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

# Import for Milvus
from pymilvus import MilvusClient, DataType

# Import local modules
from src.vector.milvus_client import MilvusConfig
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def create_milvus_collection(milvus_config):
    """Create a Milvus collection.
    
    Args:
        milvus_config: Milvus configuration
        
    Returns:
        bool: True if collection was created, False otherwise
    """
    logger.info(f"Creating Milvus collection: {milvus_config.collection_name}")
    
    try:
        # Initialize Milvus client
        client = MilvusClient(
            uri=f"http://{milvus_config.host}:{milvus_config.port}",
            token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
        )

        # Select database if needed
        if hasattr(milvus_config, 'database') and milvus_config.database:
            client.use_database(db_name=milvus_config.database)
        
        print("milvus_config.database >>>", milvus_config.database)
        # Check if collection already exists
        if client.has_collection(milvus_config.collection_name):
            logger.info(f"Collection {milvus_config.collection_name} already exists")
            return True
        
        # Create schema
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        
        # TODO DEBUG
        print("add schema fileds >>>", milvus_config.dense_vector_dim, )
        aaa = milvus_config.sparse_vector_dim if milvus_config.sparse_vector_dim else milvus_config.dense_vector_dim
        print("<<<<<<", aaa)
        # Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="cancer_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="year", datatype=DataType.VARCHAR, max_length=10)
        schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=150, nullable=True)
        schema.add_field(field_name="page_num", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=255)
        # schema.add_field(field_name="dense_embedding", datatype=DataType.FLOAT_VECTOR, dim=milvus_config.dense_vector_dim)
        # schema.add_field(field_name="sparse_embedding", datatype=DataType.FLOAT_VECTOR, dim=milvus_config.sparse_vector_dim if milvus_config.sparse_vector_dim else milvus_config.dense_vector_dim)

        # schema.add_field(field_name="dense_embedding", datatype=DataType.FLOAT_VECTOR, dim=384)
        # schema.add_field(field_name="sparse_embedding", datatype=DataType.FLOAT_VECTOR, dim=3000)
        schema.add_field(field_name="sparse_embedding", datatype=DataType.FLOAT_VECTOR, dim=3000)
        # Prepare index parameters
        index_params = client.prepare_index_params()
        
        # Add ID field index
        index_params.add_index(
            field_name="id",
            index_type="AUTOINDEX"
        )
        
        # Add vector field index
        # index_params.add_index(
        #     field_name="dense_embedding", 
        #     index_type=milvus_config.dense_index_type,
        #     # metric_type=milvus_config.dense_metric_type
        # )
        
        index_params.add_index(
            field_name="sparse_embedding", 
            index_type=milvus_config.sparse_index_type,
            # metric_type=milvus_config.sparse_metric_type
        )
        
        # Create collection
        client.create_collection(
            collection_name=milvus_config.collection_name,
            schema=schema,
            index_params=index_params
        )
        print("finish create collection success >>>")
        
        logger.info(f"Successfully created collection: {milvus_config.collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False

def load_and_store_jsonl(file_path, milvus_config, batch_size=5):
    """Load embeddings from a JSONL file and store them in Milvus.
    
    Args:
        file_path: Path to the JSONL file
        milvus_config: Milvus configuration
        batch_size: Batch size for insertion
        
    Returns:
        int: Number of inserted records
    """
    logger.info(f"Loading and storing records from {file_path} to Milvus")
    
    # Initialize Milvus client
    client = MilvusClient(
        uri=f"http://{milvus_config.host}:{milvus_config.port}",
        token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
    )
    
    # Select database if needed
    if hasattr(milvus_config, 'database') and milvus_config.database:
        client.use_database(db_name=milvus_config.database)
    
    # Ensure collection exists
    if not client.has_collection(milvus_config.collection_name):
        logger.info(f"Collection {milvus_config.collection_name} does not exist, creating...")
        create_milvus_collection(milvus_config)
    
    # Load data from JSONL file and insert in batches
    total_inserted = 0
    batch_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON object
                    record = json.loads(line)
                    
                    # Extract embedding and metadata
                    if 'embedding' in record and 'metadata' in record:
                        emb = record['embedding']
                        meta = record['metadata']
                        
                        # Prepare data record
                        data_record = {
                            "text": meta.get("text", ""),
                            "chunk_id": meta.get("chunk_id", f"chunk_{line_num}"),
                            "cancer_name": meta.get("cancer_name", ""),
                            "year": meta.get("year", ""),
                            "version": meta.get("version", ""),
                            "language": meta.get("language", ""),
                            "author": meta.get("author", ""),
                            "section": meta.get("section", ""),
                            "page_num": meta.get("page_num", 0),
                            "source_file": meta.get("source_file", "")
                        }
                        
                        # Add embedding
                        # if isinstance(emb, dict) and 'dense' in emb and 'sparse' in emb:
                        #     # Hybrid embedding
                        #     data_record["dense_embedding"] = emb['dense']
                        #     data_record["sparse_embedding"] = emb['sparse']
                        # elif isinstance(emb, list) or isinstance(emb, np.ndarray):
                        #     # Single type embedding
                        #     if milvus_config.vector_type == "dense":
                        #         data_record["dense_embedding"] = emb
                        #         # Use zero vector for sparse field
                        #         data_record["sparse_embedding"] = np.zeros(milvus_config.sparse_vector_dim if milvus_config.sparse_vector_dim else len(emb))
                        #     else:
                        #         data_record["sparse_embedding"] = emb
                        #         # Use zero vector for dense field
                        #         data_record["dense_embedding"] = np.zeros(milvus_config.dense_vector_dim if milvus_config.dense_vector_dim else len(emb))
                        
                        data_record["sparse_embedding"] = emb
                        batch_data.append(data_record)
                        
                        # Insert data if batch is full
                        if len(batch_data) >= batch_size:
                            inserted = _insert_batch(client, milvus_config.collection_name, batch_data)
                            total_inserted += inserted
                            batch_data = []
                    
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue
        
        # Insert remaining data
        if batch_data:
            inserted = _insert_batch(client, milvus_config.collection_name, batch_data)
            total_inserted += inserted
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    
    logger.info(f"Total inserted: {total_inserted} records")
    return total_inserted

def store_embeddings_in_milvus(metadata, embeddings, milvus_config, batch_size=100, reset_collection=False):
    """Store embeddings in Milvus.
    
    Args:
        metadata: List of metadata
        embeddings: List of embeddings
        milvus_config: Milvus configuration
        batch_size: Batch size for insertion
        reset_collection: Whether to reset the collection
        
    Returns:
        int: Number of inserted records
    """
    logger.info(f"Storing {len(metadata)} embeddings in Milvus collection: {milvus_config.collection_name}")
    
    # Initialize Milvus client
    client = MilvusClient(
        uri=f"http://{milvus_config.host}:{milvus_config.port}",
        token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
    )
    
    # Select database if needed
    if hasattr(milvus_config, 'database') and milvus_config.database:
        client.use_database(db_name=milvus_config.database)
    
    # Reset collection if requested
    if reset_collection:
        logger.info(f"Resetting collection {milvus_config.collection_name}")
        if client.has_collection(milvus_config.collection_name):
            client.drop_collection(milvus_config.collection_name)
        create_milvus_collection(milvus_config)
    
    # Ensure collection exists
    if not client.has_collection(milvus_config.collection_name):
        logger.info(f"Collection {milvus_config.collection_name} does not exist, creating...")
        create_milvus_collection(milvus_config)
    
    # Store embeddings in batches
    total_inserted = 0
    
    # Prepare batch data
    for i in range(0, len(metadata), batch_size):
        end_idx = min(i + batch_size, len(metadata))
        batch_metadata = metadata[i:end_idx]
        # batch_dense = embeddings[i:end_idx]
        # batch_sparse = embeddings[i:end_idx] if embeddings is not None else [np.zeros(milvus_config.sparse_vector_dim) for _ in range(end_idx - i)]
        batch_sparse = embeddings[i:end_idx]
        
        # Prepare data records
        batch_data = []
        for j, meta in enumerate(batch_metadata):
            data_record = {
                "text": meta.get("text", ""),
                "chunk_id": meta.get("chunk_id", f"chunk_{i+j}"),
                "cancer_name": meta.get("cancer_name", ""),
                "year": meta.get("year", ""),
                "version": meta.get("version", ""),
                "language": meta.get("language", ""),
                "author": meta.get("author", ""),
                "section": meta.get("section", ""),
                "page_num": meta.get("page_num", 0),
                "source_file": meta.get("source_file", ""),
                # "dense_embedding": batch_dense[j],
                "sparse_embedding": batch_sparse[j]
            }
            batch_data.append(data_record)
        
        # Insert into Milvus
        inserted = _insert_batch(client, milvus_config.collection_name, batch_data)
        total_inserted += inserted
        logger.info(f"Inserted batch {i//batch_size + 1}: {inserted} records")
    
    logger.info(f"Total inserted: {total_inserted} records")
    return total_inserted

def _insert_batch(client, collection_name, batch_data):
    """Insert a batch of data into Milvus collection.
    
    Args:
        client: Milvus client
        collection_name: Name of the collection
        batch_data: Batch data to insert
        
    Returns:
        int: Number of inserted records
    """
    try:
        # Insert data using Milvus client
        result = client.insert(
            collection_name=collection_name,
            data=batch_data
        )
        
        # Process result
        if hasattr(result, 'insert_count'):
            inserted = result.insert_count
        elif hasattr(result, 'data') and isinstance(result.data, dict) and 'insert_count' in result.data:
            inserted = result.data['insert_count']
        elif isinstance(result, dict) and 'insert_count' in result:
            inserted = result['insert_count']
        else:
            # If insert_count cannot be retrieved, assume all records were inserted
            inserted = len(batch_data)
            logger.warning(f"Unable to retrieve insert_count, assuming all {inserted} records were inserted")
        
        return inserted
    
    except Exception as e:
        logger.error(f"Failed to insert batch: {e}")
        # Add more detailed error information
        if 'result' in locals():
            logger.error(f"Insert result type: {type(result)}, content: {result}")
        return 0

def search_milvus(query_text, milvus_config, embedder, top_k=5, filter_expr=None):
    """Search for similar vectors in Milvus.
    
    Args:
        query_text: Query text
        milvus_config: Milvus configuration
        embedder: Embedder
        top_k: Number of results to return
        filter_expr: Filter expression
        
    Returns:
        list: Search results
    """
    logger.info(f"Searching for: {query_text}")
    
    # Initialize Milvus client
    client = MilvusClient(
        uri=f"http://{milvus_config.host}:{milvus_config.port}",
        token=f"{milvus_config.user}:{milvus_config.password}" if milvus_config.user and milvus_config.password else None
    )
    
    # Select database if needed
    if hasattr(milvus_config, 'database') and milvus_config.database:
        client.use_database(db_name=milvus_config.database)
    
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
    
    # Search
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
