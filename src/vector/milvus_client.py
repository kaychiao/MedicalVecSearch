#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field

# Import for Milvus
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Import setup_logger
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

@dataclass
class MilvusConfig:
    """Configuration for Milvus connection."""
    host: str = "localhost"
    port: str = "19530"
    user: str = "root"
    password: str = "Milvus"
    database: str = "default"
    collection_name: str = "nccn_guidelines"
    vector_type: Literal["dense", "sparse", "hybrid"] = "dense"
    dense_vector_dim: int = 3000  # Default for many embedding models
    sparse_vector_dim: Optional[int] = None
    dense_index_type: str = "IVF_FLAT"
    sparse_index_type: str = "FLAT"
    dense_metric_type: str = "L2"
    sparse_metric_type: str = "IP"  # Inner Product for sparse vectors
    dense_index_params: Dict[str, Any] = None
    sparse_index_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dense_index_params is None:
            self.dense_index_params = {"nlist": 1024}
        if self.sparse_index_params is None:
            self.sparse_index_params = {"nlist": 1024}
