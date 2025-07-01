#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Import centralized logging
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def save_embeddings_per_pdf(embeddings, metadata, output_dir, vector_dim=None):
    """Save embeddings and metadata to separate files per PDF.
    
    Args:
        embeddings: The embeddings data (dict for hybrid, list for single type)
        metadata: List of metadata for each embedding
        output_dir: Directory to save the embedding files
        vector_dim: Vector dimension information
    
    Returns:
        List of file paths where embeddings were saved
    """
    logger.info(f"Saving embeddings per PDF to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group embeddings by source file
    embeddings_by_file = {}
    
    # Track which index corresponds to which file
    for i, meta in enumerate(metadata):
        source_file = meta.get('source_file', 'unknown')
        if source_file not in embeddings_by_file:
            embeddings_by_file[source_file] = {
                'metadata': [],
                'indices': []
            }
        embeddings_by_file[source_file]['metadata'].append(meta)
        embeddings_by_file[source_file]['indices'].append(i)
    
    # Save embeddings for each file
    saved_files = []
    for source_file, data in embeddings_by_file.items():
        # Create filename based on source PDF
        pdf_name = Path(source_file).stem
        output_file = os.path.join(output_dir, f"{pdf_name}_embeddings.jsonl")
        
        # Extract embeddings for this file
        file_metadata = data['metadata']
        indices = data['indices']
        
        if isinstance(embeddings, dict):  # Hybrid embeddings
            file_embeddings = {
                'dense': [embeddings['dense'][i] for i in indices],
                'sparse': [embeddings['sparse'][i] for i in indices]
            }
        else:  # Single type embeddings
            file_embeddings = [embeddings[i] for i in indices]
        
        # Save to JSONL file (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, meta in enumerate(file_metadata):
                # Prepare embedding data
                if isinstance(file_embeddings, dict):  # Hybrid embeddings
                    emb_data = {
                        'dense': file_embeddings['dense'][i].tolist() if isinstance(file_embeddings['dense'][i], np.ndarray) else file_embeddings['dense'][i],
                        'sparse': file_embeddings['sparse'][i].tolist() if isinstance(file_embeddings['sparse'][i], np.ndarray) else file_embeddings['sparse'][i]
                    }
                else:  # Single type embeddings
                    emb_data = file_embeddings[i].tolist() if isinstance(file_embeddings[i], np.ndarray) else file_embeddings[i]
                
                # Create record
                record = {
                    'embedding': emb_data,
                    'metadata': meta,
                    'vector_dim': vector_dim
                }
                
                # Write as a single line
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(file_metadata)} embeddings to {output_file}")
        saved_files.append(output_file)
    
    return saved_files

def load_embeddings_from_jsonl(file_path):
    """Load embeddings and metadata from a JSONL file where each line is a JSON object.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Tuple of (embeddings, metadata, vector_dim)
    """
    logger.info(f"Loading embeddings from JSONL file: {file_path}")
    
    embeddings = []
    metadata = []
    vector_dim = None
    embedding_type = None
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return [], [], None
        
        # Read file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON object from line
                    record = json.loads(line)
                    
                    # Extract embedding and metadata
                    if 'embedding' in record and 'metadata' in record:
                        emb = record['embedding']
                        
                        # # Determine embedding type (only on first record)
                        # if embedding_type is None:
                        #     if isinstance(emb, dict) and 'dense' in emb and 'sparse' in emb:
                        #         embedding_type = 'hybrid'
                        #         # Initialize embeddings dict for hybrid
                        #         if not embeddings:
                        #             embeddings = {'dense': [], 'sparse': []}
                        #         # Get vector dimensions if not already set
                        #         if vector_dim is None and 'vector_dim' in record:
                        #             vector_dim = record['vector_dim']
                        #     else:
                        #         embedding_type = 'single'
                        
                        # # Add embedding to the appropriate structure
                        # if embedding_type == 'hybrid':
                        #     # Convert to numpy arrays
                        #     embeddings['dense'].append(np.array(emb['dense']))
                        #     embeddings['sparse'].append(np.array(emb['sparse']))
                        # else:
                        #     embeddings.append(np.array(emb))
                        
                        embeddings.append(np.array(emb))
                        # Add metadata
                        metadata.append(record['metadata'])
                
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error opening or reading file {file_path}: {e}")
        return [], [], None
    
    logger.info(f"Loaded {len(metadata)} records from {file_path}")
    return embeddings, metadata, vector_dim
