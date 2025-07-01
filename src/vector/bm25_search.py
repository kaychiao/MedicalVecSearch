#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Import for Milvus
from pymilvus import MilvusClient, DataType, Function, FunctionType

# Import centralized logging
from src.utils.logging_config import setup_logger
from src.vector.milvus_client import MilvusConfig

# Set up logger
logger = setup_logger(__name__)

class BM25Config(MilvusConfig):
    """Configuration for BM25 full-text search."""
    def __init__(self, 
                 host: str = "10.10.163.183",
                 port: str = "19530",
                 user: str = "root",
                 password: str = "Milvus",
                 database: str = "nccn_bm25",
                 collection_name: str = "nccn_guidelines_bm25",
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.75,
                 inverted_index_algo: str = "DAAT_MAXSCORE"):
        """Initialize BM25Config with default values."""
        super().__init__(host=host, port=port, user=user, password=password, 
                         database=database, collection_name=collection_name)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.inverted_index_algo = inverted_index_algo


def create_bm25_collection(bm25_config: BM25Config) -> bool:
    """Create a Milvus collection for BM25 full-text search.
    
    Args:
        bm25_config: BM25 configuration
        
    Returns:
        bool: True if collection was created, False otherwise
    """
    logger.info(f"Creating BM25 full-text search collection: {bm25_config.collection_name}")
    
    try:
        # Initialize Milvus client
        client = MilvusClient(
            uri=f"http://{bm25_config.host}:{bm25_config.port}",
            token=f"{bm25_config.user}:{bm25_config.password}" if bm25_config.user and bm25_config.password else None
        )

        # Select database if needed
        if hasattr(bm25_config, 'database') and bm25_config.database:
            client.use_database(db_name=bm25_config.database)
        
        # Check if collection already exists
        if client.has_collection(bm25_config.collection_name):
            logger.info(f"Collection {bm25_config.collection_name} already exists")
            return True
        
        # Create schema
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        
        # Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="cancer_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="year", datatype=DataType.VARCHAR, max_length=10)
        schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=150, nullable=True)
        schema.add_field(field_name="page_num", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        
        # Define BM25 function
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25
        )
        
        # Add function to schema
        schema.add_function(bm25_function)
        
        # Prepare index parameters
        index_params = client.prepare_index_params()
        
        # Add ID field index
        index_params.add_index(
            field_name="id",
            index_type="AUTOINDEX"
        )
        
        # Add sparse vector field index for BM25
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": bm25_config.inverted_index_algo,
                "bm25_k1": bm25_config.bm25_k1,
                "bm25_b": bm25_config.bm25_b
            }
        )
        
        # Create collection
        client.create_collection(
            collection_name=bm25_config.collection_name,
            schema=schema,
            index_params=index_params
        )
        
        logger.info(f"Successfully created BM25 collection: {bm25_config.collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create BM25 collection: {e}")
        return False


def load_and_insert_chunks(chunks_dir: str, bm25_config: BM25Config, batch_size: int = 100) -> bool:
    """Load chunks from JSON files and insert them into the BM25 collection.
    
    Args:
        chunks_dir: Directory containing chunk JSON files
        bm25_config: BM25 configuration
        batch_size: Number of chunks to insert in each batch
        
    Returns:
        bool: True if data was inserted successfully, False otherwise
    """
    logger.info(f"Loading and inserting chunks from {chunks_dir} into BM25 collection")
    
    try:
        # Initialize Milvus client
        client = MilvusClient(
            uri=f"http://{bm25_config.host}:{bm25_config.port}",
            token=f"{bm25_config.user}:{bm25_config.password}" if bm25_config.user and bm25_config.password else None
        )

        # Select database if needed
        if hasattr(bm25_config, 'database') and bm25_config.database:
            client.use_database(db_name=bm25_config.database)
        
        # Check if collection exists
        if not client.has_collection(bm25_config.collection_name):
            logger.error(f"Collection {bm25_config.collection_name} does not exist")
            return False
        
        # Get all chunk files in the directory
        chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('_chunks.json')]
        
        if not chunk_files:
            logger.warning(f"No chunk files found in {chunks_dir}")
            return False
        
        total_inserted = 0
        
        # Process each chunk file
        for chunk_file in chunk_files:
            file_path = os.path.join(chunks_dir, chunk_file)
            logger.info(f"Processing chunk file: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                # Insert chunks in batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    
                    # Prepare data for insertion
                    insert_data = []
                    for chunk in batch:
                        # Handle page_num field (convert to int or None)
                        page_num = None
                        if chunk.get('page_num') is not None:
                            try:
                                page_num = int(chunk['page_num'])
                            except (ValueError, TypeError):
                                page_num = None
                        
                        # Create data entry with all required fields
                        entry = {
                            'text': chunk.get('text', ''),
                            'chunk_id': chunk.get('chunk_id', ''),
                            'cancer_name': chunk.get('cancer_name', ''),
                            'year': chunk.get('year', ''),
                            'version': chunk.get('version', ''),
                            'language': chunk.get('language', ''),
                            'author': chunk.get('author', ''),
                            'section': chunk.get('section', ''),
                            'page_num': page_num,
                            'source_file': chunk.get('source_file', '')
                        }
                        insert_data.append(entry)
                    
                    # Insert batch
                    if insert_data:
                        try:
                            result = client.insert(
                                collection_name=bm25_config.collection_name,
                                data=insert_data
                            )
                            
                            # Check if result has insert_count attribute
                            if hasattr(result, 'insert_count'):
                                batch_inserted = result.insert_count
                            else:
                                # For newer Milvus versions, the result might be a dictionary
                                batch_inserted = result.get('insert_count', len(insert_data))
                                if not isinstance(batch_inserted, int):
                                    # If we still can't get the count, use the batch size as an estimate
                                    batch_inserted = len(insert_data)
                            
                            total_inserted += batch_inserted
                            logger.info(f"Inserted {batch_inserted} chunks from {chunk_file} (batch {i//batch_size + 1})")
                        except Exception as e:
                            logger.error(f"Error inserting batch from {chunk_file}: {e}")
                            # Continue with the next batch instead of failing the entire file
                            continue
            
            except Exception as e:
                logger.error(f"Error processing chunk file {chunk_file}: {e}")
                continue
        
        logger.info(f"Total chunks inserted into BM25 collection: {total_inserted}")
        return total_inserted > 0
        
    except Exception as e:
        logger.error(f"Failed to insert chunks into BM25 collection: {e}")
        return False


def search_bm25(query: str, bm25_config: BM25Config, limit: int = 10, output_fields: List[str] = None) -> List[Dict]:
    """Perform BM25 full-text search on the collection.
    
    Args:
        query: Search query text
        bm25_config: BM25 configuration
        limit: Maximum number of results to return
        output_fields: Fields to include in the results (if None, all fields are returned)
        
    Returns:
        List of search results
    """
    logger.info(f"Performing BM25 search with query: '{query}'")
    
    try:
        # Initialize Milvus client
        client = MilvusClient(
            uri=f"http://{bm25_config.host}:{bm25_config.port}",
            token=f"{bm25_config.user}:{bm25_config.password}" if bm25_config.user and bm25_config.password else None
        )

        # Select database if needed
        if hasattr(bm25_config, 'database') and bm25_config.database:
            client.use_database(db_name=bm25_config.database)
        
        # Check if collection exists
        if not client.has_collection(bm25_config.collection_name):
            logger.error(f"Collection {bm25_config.collection_name} does not exist")
            return []
        
        # Define default output fields if not provided
        if output_fields is None:
            output_fields = ["text", "chunk_id", "cancer_name", "year", "version", 
                             "language", "author", "section", "page_num", "source_file"]
        
        # Define search parameters
        search_params = {
            'params': {'drop_ratio_search': 0.2},
        }
        
        # Perform search
        results = client.search(
            collection_name=bm25_config.collection_name,
            data=[query],
            anns_field="sparse",
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )
        
        # Format results
        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                result = {
                    'score': hit.score,
                    'id': hit.id
                }
                print("<<<<< hit :", hit)
                # Add output fields
                for field in output_fields:
                    if field in hit.entity:
                        result[field] = hit.entity[field]
                
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Failed to perform BM25 search: {e}")
        return []
