#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import centralized logging
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def save_chunks_to_file(chunks, file_path):
    """Save chunks to a JSON file.
    
    Args:
        chunks: List of text chunks
        file_path: Path to save the chunks
    """
    logger.info(f"Saving {len(chunks)} chunks to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert chunks to dictionaries
    chunks_data = [chunk.to_dict() for chunk in chunks]
    
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to {file_path}")

def load_chunks_from_file(file_path):
    """Load chunks from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of chunk objects
    """
    logger.info(f"Loading chunks from {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Load from file
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Convert dictionaries to chunk objects
        from src.pdf.processor import TextChunk
        from src.pdf.metadata import NCCNMetadata
        
        chunks = []
        for chunk_data in chunks_data:
            # print("chunk_data >>>", chunk_data)
            # Extract metadata fields from the chunk data
            metadata_obj = NCCNMetadata(
                filename=chunk_data.get('source_file', ''),
                cancer_name=chunk_data.get('cancer_name', 'Unknown'),
                year=chunk_data.get('year'),
                version=chunk_data.get('version'),
                language=chunk_data.get('language', 'English'),
                author=chunk_data.get('author', 'NCCN')
            )
            
            # Create TextChunk with proper metadata object
            chunk = TextChunk(
                text=chunk_data.get('text', ''),
                metadata=metadata_obj,
                section=chunk_data.get('section'),
                page_num=chunk_data.get('page_num'),
                chunk_id=chunk_data.get('chunk_id')
            )
            chunks.append(chunk)
        
        logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    
    except Exception as e:
        logger.error(f"Error loading chunks from {file_path}: {e}")
        return []
