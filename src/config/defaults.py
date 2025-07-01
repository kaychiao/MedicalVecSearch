#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
默认配置模块，定义所有可配置参数的默认值。
这些值可以被环境变量或命令行参数覆盖。
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class AppConfig:
    """应用程序配置类，包含所有可配置参数。"""
    
    # 输入/输出路径
    input: Optional[str] = None
    models: Optional[str] = None
    intermediate_dir: Optional[str] = None
    
    # 目录路径，而不是具体文件
    chunks_dir: Optional[str] = None
    embeddings_dir: Optional[str] = None
    jsonl_file: Optional[str] = None  # 这个仍然是文件，因为它是直接加载到Milvus的
    
    # 处理步骤
    steps: str = "extract,generate_embeddings,store_in_milvus"
    start_from: Optional[str] = None
    
    # 分块参数
    chunk_size: int = 6000
    chunk_overlap: int = 20
    
    # 内存管理参数
    batch_size: int = 1
    save_intermediate: bool = True
    
    # GPU支持
    use_gpu: bool = True
    
    # 嵌入模型参数
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_type: str = "dense"  # 'dense', 'sparse', 或 'hybrid'
    sparse_model: str = "distilbert-base-uncased"
    dense_vector_dim: Optional[int] = None
    sparse_vector_dim: Optional[int] = None
    
    # Milvus连接参数
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_user: str = "root"
    milvus_password: str = "Milvus"
    milvus_database: str = "default"
    collection_name: str = "medical_documents"
    reset_collection: bool = False
    
    def update_from_env(self):
        """从环境变量更新配置。"""
        # 输入/输出路径
        self.input = os.getenv("NCCN_INPUT", self.input)
        self.models = os.getenv("NCCN_MODELS", self.models)
        self.intermediate_dir = os.getenv("NCCN_INTERMEDIATE_DIR", self.intermediate_dir)
        
        # 如果设置了中间目录，但没有设置其他目录路径，则使用默认路径
        if self.intermediate_dir:
            # 只有在环境变量中没有明确设置时，才使用默认路径
            self.chunks_dir = os.getenv("NCCN_CHUNKS_DIR", 
                                       os.path.join(self.intermediate_dir, "chunks"))
            self.embeddings_dir = os.getenv("NCCN_EMBEDDINGS_DIR", 
                                           os.path.join(self.intermediate_dir, "embeddings"))
            # jsonl_file仍然是一个文件路径，因为它是直接加载到Milvus的
            self.jsonl_file = os.getenv("NCCN_JSONL_FILE", self.jsonl_file)
        else:
            # 如果没有设置中间目录，则直接从环境变量获取
            self.chunks_dir = os.getenv("NCCN_CHUNKS_DIR", self.chunks_dir)
            self.embeddings_dir = os.getenv("NCCN_EMBEDDINGS_DIR", self.embeddings_dir)
            self.jsonl_file = os.getenv("NCCN_JSONL_FILE", self.jsonl_file)
        
        # 处理步骤
        self.steps = os.getenv("NCCN_STEPS", self.steps)
        self.start_from = os.getenv("NCCN_START_FROM", self.start_from)
        
        # 分块参数
        chunk_size_env = os.getenv("NCCN_CHUNK_SIZE")
        if chunk_size_env:
            self.chunk_size = int(chunk_size_env)
            
        chunk_overlap_env = os.getenv("NCCN_CHUNK_OVERLAP")
        if chunk_overlap_env:
            self.chunk_overlap = int(chunk_overlap_env)
        
        # 内存管理参数
        batch_size_env = os.getenv("NCCN_BATCH_SIZE")
        if batch_size_env:
            self.batch_size = int(batch_size_env)
            
        self.save_intermediate = os.getenv("NCCN_SAVE_INTERMEDIATE", str(self.save_intermediate)).lower() == "true"
        
        # GPU支持
        self.use_gpu = os.getenv("NCCN_USE_GPU", str(self.use_gpu)).lower() == "true"
        
        # 嵌入模型参数
        self.embedding_model = os.getenv("NCCN_EMBEDDING_MODEL", self.embedding_model)
        self.embedding_type = os.getenv("NCCN_EMBEDDING_TYPE", self.embedding_type)
        self.sparse_model = os.getenv("NCCN_SPARSE_MODEL", self.sparse_model)
        
        dense_dim_env = os.getenv("NCCN_DENSE_VECTOR_DIM")
        if dense_dim_env:
            self.dense_vector_dim = int(dense_dim_env)
            
        sparse_dim_env = os.getenv("NCCN_SPARSE_VECTOR_DIM")
        if sparse_dim_env:
            self.sparse_vector_dim = int(sparse_dim_env)
        
        # Milvus连接参数
        self.milvus_host = os.getenv("MILVUS_HOST", self.milvus_host)
        self.milvus_port = os.getenv("MILVUS_PORT", self.milvus_port)
        self.milvus_user = os.getenv("MILVUS_USER", self.milvus_user)
        self.milvus_password = os.getenv("MILVUS_PASSWORD", self.milvus_password)
        self.milvus_database = os.getenv("MILVUS_DATABASE", self.milvus_database)
        self.collection_name = os.getenv("MILVUS_COLLECTION", self.collection_name)
        self.reset_collection = os.getenv("MILVUS_RESET_COLLECTION", str(self.reset_collection)).lower() == "true"
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置。"""
        for key, value in config_dict.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    
    def update_from_args(self, args):
        """从命令行参数更新配置。"""
        # 将args转换为字典
        args_dict = vars(args)
        
        # 特殊处理no_开头的参数
        if 'no_save_intermediate' in args_dict and args_dict['no_save_intermediate']:
            args_dict['save_intermediate'] = False
            del args_dict['no_save_intermediate']
            
        if 'no_gpu' in args_dict and args_dict['no_gpu']:
            args_dict['use_gpu'] = False
            del args_dict['no_gpu']
        
        # 特殊处理文件路径参数
        if 'chunks_file' in args_dict and args_dict['chunks_file']:
            # 如果指定了chunks_file，我们将其目录作为chunks_dir
            self.chunks_dir = os.path.dirname(args_dict['chunks_file'])
            del args_dict['chunks_file']
            
        if 'embeddings_file' in args_dict and args_dict['embeddings_file']:
            # 如果指定了embeddings_file，我们将其目录作为embeddings_dir
            self.embeddings_dir = os.path.dirname(args_dict['embeddings_file'])
            del args_dict['embeddings_file']
        
        self.update_from_dict(args_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典。"""
        return {
            # 输入/输出路径
            "input": self.input,
            "models": self.models,
            "intermediate_dir": self.intermediate_dir,
            "chunks_dir": self.chunks_dir,
            "embeddings_dir": self.embeddings_dir,
            "jsonl_file": self.jsonl_file,
            
            # 处理步骤
            "steps": self.steps,
            "start_from": self.start_from,
            
            # 分块参数
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            
            # 内存管理参数
            "batch_size": self.batch_size,
            "save_intermediate": self.save_intermediate,
            
            # GPU支持
            "use_gpu": self.use_gpu,
            
            # 嵌入模型参数
            "embedding_model": self.embedding_model,
            "embedding_type": self.embedding_type,
            "sparse_model": self.sparse_model,
            "dense_vector_dim": self.dense_vector_dim,
            "sparse_vector_dim": self.sparse_vector_dim,
            
            # Milvus连接参数
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "milvus_user": self.milvus_user,
            "milvus_password": self.milvus_password,
            "milvus_database": self.milvus_database,
            "collection_name": self.collection_name,
            "reset_collection": self.reset_collection,
        }

def load_config() -> AppConfig:
    """加载配置，优先级：默认值 < 环境变量"""
    config = AppConfig()
    config.update_from_env()
    return config
