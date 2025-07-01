#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize

# Import centralized logging
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

class VectorEmbedder:
    """Generate vector embeddings from text using a model."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 embedding_type: Literal["dense", "sparse", "hybrid"] = "dense",
                 sparse_model_name: Optional[str] = "distilbert-base-uncased",
                 dense_vector_dim: Optional[int] = None,
                 sparse_vector_dim: Optional[int] = None):
        """Initialize with specific model(s).
        
        Args:
            model_name: The name of the dense embedding model
            embedding_type: Type of embedding to generate - "dense", "sparse", or "hybrid"
            sparse_model_name: The name of the sparse embedding model (if needed)
            dense_vector_dim: Optional custom dimension for dense vectors
            sparse_vector_dim: Optional custom dimension for sparse vectors
        """
        logger.info(f"Initializing embedder with type: {embedding_type}")
        self.embedding_type = embedding_type
        
        # Dense model setup
        if embedding_type in ["dense", "hybrid"]:
            logger.info(f"Loading dense embedding model: {model_name}")
            self.dense_model_name = model_name
            self.dense_model = SentenceTransformer(model_name)
            self.original_dense_dim = self.dense_model.get_sentence_embedding_dimension()
            
            # Use custom dimension or original dimension
            self.dense_vector_dim = dense_vector_dim or self.original_dense_dim
            if dense_vector_dim and dense_vector_dim != self.original_dense_dim:
                logger.info(f"Using custom dense dimension: {dense_vector_dim} (original: {self.original_dense_dim})")
            else:
                logger.info(f"Dense model loaded with dimension: {self.dense_vector_dim}")
        
        # Sparse model setup
        if embedding_type in ["sparse", "hybrid"]:
            logger.info(f"Loading sparse embedding model: {sparse_model_name}")
            self.sparse_model_name = sparse_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(sparse_model_name)
            self.sparse_model = AutoModel.from_pretrained(sparse_model_name)
            # Vocabulary size determines the sparse vector dimension
            self.original_sparse_dim = len(self.tokenizer.vocab)
            
            # Use custom dimension or original dimension
            self.sparse_vector_dim = sparse_vector_dim or self.original_sparse_dim
            if sparse_vector_dim and sparse_vector_dim != self.original_sparse_dim:
                logger.info(f"Using custom sparse dimension: {sparse_vector_dim} (original: {self.original_sparse_dim})")
            else:
                logger.info(f"Sparse model loaded with dimension: {self.sparse_vector_dim}")
    
    def embed_text(self, text: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        """Generate embedding for a single text.
        
        Returns:
            - For dense: a dense vector
            - For sparse: a sparse vector
            - For hybrid: a tuple of (dense_vector, sparse_vector)
            - Or a dictionary with keys 'dense' and/or 'sparse'
        """
        if self.embedding_type == "dense":
            vector = self.dense_model.encode(text)
            # Adjust dimension (if needed)
            return self._resize_vector(vector, self.dense_vector_dim)
        elif self.embedding_type == "sparse":
            vector = self._generate_sparse_embedding(text)
            # Adjust dimension (if needed)
            return self._resize_vector(vector, self.sparse_vector_dim)
        else:  # hybrid
            dense_vector = self.dense_model.encode(text)
            sparse_vector = self._generate_sparse_embedding(text)
            # Adjust dimension (if needed)
            return {
                "dense": self._resize_vector(dense_vector, self.dense_vector_dim),
                "sparse": self._resize_vector(sparse_vector, self.sparse_vector_dim)
            }
    
    def embed_texts(self, texts: List[str]) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """Generate embeddings for multiple texts."""
        if self.embedding_type == "dense":
            vectors = self.dense_model.encode(texts)
            # Adjust dimension for each vector (if needed)
            return np.array([self._resize_vector(vector, self.dense_vector_dim) for vector in vectors])
        elif self.embedding_type == "sparse":
            vectors = [self._generate_sparse_embedding(text) for text in texts]
            # Adjust dimension for each vector (if needed)
            return [self._resize_vector(vector, self.sparse_vector_dim) for vector in vectors]
        else:  # hybrid
            dense_vectors = self.dense_model.encode(texts)
            sparse_vectors = [self._generate_sparse_embedding(text) for text in texts]
            # Adjust dimension for each vector (if needed)
            return {
                "dense": np.array([self._resize_vector(vector, self.dense_vector_dim) for vector in dense_vectors]),
                "sparse": [self._resize_vector(vector, self.sparse_vector_dim) for vector in sparse_vectors]
            }
    
    def _generate_sparse_embedding(self, text: str) -> np.ndarray:
        """Generate a sparse embedding for a text using token weights."""
        # Tokenize and get model output
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.sparse_model(**inputs)
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Create a sparse vector based on token weights
        # Initialize a zero vector with vocab size
        sparse_vector = np.zeros(self.sparse_vector_dim)
        
        # Get token IDs and their weights from the model output
        token_ids = inputs.input_ids[0].numpy()
        # Use the mean of the hidden states as weights
        token_weights = torch.mean(last_hidden_state[0], dim=1).numpy()
        
        # Populate the sparse vector with weights at token positions
        for token_id, weight in zip(token_ids, token_weights):
            if token_id < self.sparse_vector_dim:  # Ensure token_id is within bounds
                sparse_vector[token_id] = weight
        
        # Normalize the sparse vector
        if np.sum(sparse_vector) > 0:
            sparse_vector = normalize(sparse_vector.reshape(1, -1))[0]
        
        return sparse_vector
    
    def _resize_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Adjust vector dimension to target dimension.
        
        If target dimension is larger than original, pad with zeros.
        If target dimension is smaller than original, truncate.
        If dimensions are the same, return unchanged.
        
        Args:
            vector: Original vector
            target_dim: Target dimension
            
        Returns:
            Adjusted vector
        """
        if len(vector) == target_dim:
            return vector
            
        if len(vector) < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:len(vector)] = vector
            return padded
        else:
            # Truncate
            return vector[:target_dim]
    
    def embed_chunks(self, chunks: List) -> Union[Tuple[List[np.ndarray], List[Dict[str, Any]]], 
                                                 Dict[str, Union[List[np.ndarray], List[Dict[str, Any]]]]]:
        """Generate embeddings for text chunks and prepare metadata.
        
        The chunks should have a text attribute and a to_dict method.
        
        Returns:
            For dense/sparse: (embeddings, metadata)
            For hybrid: {
                'dense': (dense_embeddings, metadata),
                'sparse': (sparse_embeddings, metadata)
            }
        """
        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.to_dict() for chunk in chunks]
        
        if self.embedding_type == "dense":
            embeddings = self.embed_texts(texts)
            return embeddings, metadata
        elif self.embedding_type == "sparse":
            embeddings = self.embed_texts(texts)
            return embeddings, metadata
        else:  # hybrid
            result = self.embed_texts(texts)
            return {
                'dense': (result['dense'], metadata),
                'sparse': (result['sparse'], metadata)
            }
    
    @property
    def vector_dim(self) -> Union[int, Dict[str, int]]:
        """Get the vector dimension(s) of the model(s)."""
        if self.embedding_type == "dense":
            return self.dense_vector_dim
        elif self.embedding_type == "sparse":
            return self.sparse_vector_dim
        else:  # hybrid
            return {
                "dense": self.dense_vector_dim,
                "sparse": self.sparse_vector_dim
            }
