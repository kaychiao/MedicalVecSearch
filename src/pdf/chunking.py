#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from typing import List, Tuple, Optional, Generator, Iterator
import gc

# Import setup_logger
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def split_by_section(text: str) -> List[Tuple[Optional[str], str]]:
    """Split text into sections based on markdown headers.
    
    Returns a list of (section_title, section_content) tuples.
    """
    # Split by markdown headers (# Header)
    sections = []
    current_section = None
    current_content = []
    
    # 使用迭代器而不是一次性分割整个文本
    for line in text.splitlines():
        if line.startswith('#'):
            # If we have content from a previous section, add it
            if current_content:
                # 使用join前先计算大致内容大小，如果过大则进行警告
                content_size = sum(len(line) for line in current_content)
                if content_size > 10_000_000:  # 10MB警告阈值
                    logger.warning(f"Very large section detected: {content_size} bytes")
                
                sections.append((current_section, '\n'.join(current_content).strip()))
                current_content = []
                # 主动释放内存
                gc.collect()
            
            # Set new section title (remove # and whitespace)
            current_section = line.lstrip('#').strip()
        else:
            current_content.append(line)
    
    # Add the last section if there's content
    if current_content:
        sections.append((current_section, '\n'.join(current_content).strip()))
    
    return sections

def split_text_by_size_generator(text: str, max_chunk_size: int = 1000, overlap: int = 10) -> Iterator[str]:
    """Split text into chunks using a generator to minimize memory usage.
    
    Args:
        text: The text to split into chunks
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Iterator yielding text chunks
    """
    # 记录文本总长度
    text_size = len(text)
    
    # 如果文本长度小于max_chunk_size，直接返回整个文本
    if text_size <= max_chunk_size:
        yield text.strip()
        return
    
    # 初始化起始位置
    start = 0
    
    # 编译正则表达式以提高性能
    sentence_end_pattern = re.compile(r'[.!?]\s')
    
    # 防止无限循环的安全计数器
    safety_counter = 0
    max_iterations = text_size  # 设置一个合理的最大迭代次数（文本长度）
    
    # 循环直到处理完整个文本
    while start < text_size and safety_counter < max_iterations:
        safety_counter += 1
        
        # 计算当前块的结束位置
        end = min(start + max_chunk_size, text_size)
        
        # 如果不是文本末尾，尝试找到句子结束的位置
        if end < text_size:
            # 获取当前处理的文本片段，向后多看100个字符
            look_ahead = min(end + 100, text_size)
            current_text = text[start:look_ahead]
            
            # 在当前块内查找最后一个句子结束标记
            matches = list(sentence_end_pattern.finditer(current_text))
            if matches:
                # 找到最后一个在max_chunk_size范围内的句子结束位置
                valid_matches = [m for m in matches if start + m.end() <= start + max_chunk_size]
                if valid_matches:
                    # 使用找到的句子结束位置作为当前块的结束位置
                    last_match = valid_matches[-1]
                    end = start + last_match.end()
        
        # 生成当前块
        current_chunk = text[start:end].strip()
        if current_chunk:  # 确保不产生空块
            yield current_chunk
            
            # 如果已经处理到文本末尾，退出循环
            if end >= text_size:
                break
        
        # 计算下一个块的起始位置，考虑重叠
        new_start = end - overlap
        
        # 确保起始位置始终前进，防止无限循环
        if new_start <= start:
            new_start = start + 1
        
        start = min(new_start, text_size)
        
        # 主动进行垃圾回收
        gc.collect()
    
    # 如果因为安全计数器退出循环，记录警告
    if safety_counter >= max_iterations:
        logger.warning(f"Split text generator reached maximum iterations ({max_iterations}), possible infinite loop detected")
