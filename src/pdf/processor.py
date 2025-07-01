#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import datetime
import json

# Import Docling for PDF parsing
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption, FormatOption
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        )
    from docling_core.types.doc import ImageRefMode
    from docling_core.transforms.chunker import HierarchicalChunker

except ImportError:
    raise ImportError("Docling is required. Install it using 'pip install docling'")

from langchain.text_splitter import MarkdownTextSplitter
# Import local modules
from src.pdf.metadata import NCCNMetadata, extract_metadata_from_filename
from src.pdf.chunking import split_by_section, split_text_by_size_generator
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

@dataclass
class TextChunk:
    """A chunk of text from a document with its metadata."""
    text: str
    metadata: NCCNMetadata
    section: Optional[str] = None
    page_num: Optional[int] = None
    chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "text": self.text,
            "cancer_name": self.metadata.cancer_name,
            "year": self.metadata.year,
            "version": self.metadata.version,
            "language": self.metadata.language,
            "author": self.metadata.author,
            "section": self.section,
            "page_num": self.page_num,
            "chunk_id": self.chunk_id,
            "source_file": self.metadata.filename
        }

def output_file_path(pdf_path: Path, output_dir: Optional[Path] = None, organized_dirs: Optional[Dict[str, Path]] = None):
    # Save intermediate results if output_dir is provided
    output_dir = output_dir if output_dir else Path(".")
    # Generate output filenames based on input filename
    base_name = pdf_path.stem
    
    # Determine output paths based on organized_dirs if provided
    if organized_dirs and isinstance(organized_dirs, dict):
            # Use organized directory structure
            text_dir = organized_dirs.get("text", output_dir)
            markdown_dir = organized_dirs.get("markdown", output_dir)
            metadata_dir = organized_dirs.get("metadata", output_dir)
            
            text_output_path = Path(text_dir) / f"{base_name}.txt"
            markdown_output_path = Path(markdown_dir) / f"{base_name}.md"
            metadata_output_path = Path(metadata_dir) / f"{base_name}_info.json"
    else:
        text_dir = Path(output_dir) / "text"
        markdown_dir = Path(output_dir) / "markdown"
        metadata_dir = Path(output_dir) / "metadata"
        text_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Use flat directory structure
        text_output_path = Path(text_dir) / f"{base_name}.txt"
        markdown_output_path = Path(markdown_dir) / f"{base_name}.md"
        metadata_output_path = Path(metadata_dir) / f"{base_name}_info.json"

    return text_output_path, markdown_output_path, metadata_output_path

def convert_pdf_to_text(pdf_path: Path, models_dir: Optional[Path] = None, output_dir: Optional[Path] = None, organized_dirs: Optional[Dict[str, Path]] = None, use_gpu: bool = True) -> Tuple[str, str]:
    """Convert PDF to both text and markdown using Docling.
    
    Args:
        pdf_path: Path to the PDF file
        models_dir: Optional path to Docling model artifacts
        output_dir: Optional directory to save intermediate recognition results
        organized_dirs: Optional dictionary with organized subdirectories for different file types
        use_gpu: Whether to use GPU for processing (if available)
        
    Returns:
        Tuple of (text_content, markdown_content)
    """
    logger.info(f"Converting PDF: {pdf_path}")
    try:
        # Set custom models directory if provided
        if models_dir:
            # Ensure models_dir is a Path object
            if isinstance(models_dir, str):
                models_dir = Path(models_dir)
        else:
            models_dir = Path("~/.cache/docling/models").expanduser()
        
        logger.info(f"Using custom model path: {models_dir}")
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(models_dir)

        acceler_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
        
        # Ensure pdf_path is a Path object
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)
        
        pipeline_options = PdfPipelineOptions(
            artifacts_path=str(models_dir), 
            use_gpu=use_gpu, do_ocr=True, languages=['en', 'zh'], 
            accelerator_options=acceler_options
            )
        # save markdown with images
        pipeline_options.images_scale = 1.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        # Create converter with default settings
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        
        # Convert PDF
        logger.info(f"Starting conversion of: {pdf_path}")
        result = converter.convert(pdf_path)
        logger.info("Conversion complete")
        text_output_path, markdown_output_path, metadata_output_path = output_file_path(pdf_path, output_dir, organized_dirs)
        
        # Save markdown content
        result.document.save_as_markdown(markdown_output_path, image_mode=ImageRefMode.REFERENCED, delim="\n")
        with open(markdown_output_path, 'r', encoding='utf-8') as f:
            markdown_length = sum(len(line) for line in f)
        logger.info(f"Saved markdown content ({markdown_length} chars) to {markdown_output_path}")

        
        text_content = result.document.export_to_text(delim="\n") if hasattr(result.document, 'export_to_text') else ""
        if not text_content:
            logger.info("Extracting plain text from markdown content")
            markdown_content = result.document.export_to_markdown(delim="\n", image_mode=ImageRefMode.REFERENCED)
            # Simple markdown to text conversion (remove headers, etc.)
            text_content = '\n'.join([
                line for line in markdown_content.split('\n') 
                if not line.startswith('#') and line.strip()
            ])
        # Save text content
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
            text_length = len(text_content)
        logger.info(f"Saved plain text content ({text_length} chars) to {text_output_path}")
        del text_content
        gc.collect()
   
        # Save metadata about the extraction
        extraction_info = {
            "source_file": str(pdf_path),
            "extraction_time": str(datetime.datetime.now()),
            "text_size_bytes": text_length,
            "markdown_size_bytes": markdown_length,
            "text_output_file": str(text_output_path),
            "markdown_output_file": str(markdown_output_path)
        }
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_info, f, indent=2)
        logger.info(f"Saved extraction metadata to {metadata_output_path}")
        
        return text_output_path, markdown_output_path
    except Exception as e:
        logger.error(f"Error converting PDF: {pdf_path}, Error: {e}")
        return "", ""

def chunking_with_langchain(source, max_chunk_size, chunk_overlap):
    with open(source, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    splitter = MarkdownTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(markdown_content)
    return chunks

def process_pdf_to_chunks(
    pdf_path: Path, 
    models_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    organized_dirs: Optional[Dict[str, Path]] = None,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 100,
    use_gpu: bool = True
) -> List[TextChunk]:
    """Process a PDF file and return a list of text chunks with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        models_dir: Optional path to Docling model artifacts
        output_dir: Optional directory to save intermediate results
        organized_dirs: Optional dictionary with organized subdirectories
        max_chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between chunks
        use_gpu: Whether to use GPU for processing (if available)
        
    Returns:
        List of TextChunk objects with extracted content
    """
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(pdf_path)
    
    # Convert PDF to text and markdown
    _, markdown_output_path = convert_pdf_to_text(pdf_path, models_dir, output_dir, organized_dirs, use_gpu=use_gpu)
    
    results = []

    # Convert the input file to Docling Document
    chunks = chunking_with_langchain(markdown_output_path, max_chunk_size, chunk_overlap)

    # Perform hierarchical chunking
    for chunk_id, chunk in enumerate(chunks):
        results.append(TextChunk(
            text=chunk,
            metadata=metadata,
            section="chunk.section",
            chunk_id=f"{metadata.cancer_name}_{chunk_id}"
        ))
    logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
    return results

def find_pdf_files(directory: Path) -> List[Path]:
    """Find all PDF files in the given directory."""
    # Convert string to Path object if it's a string
    if isinstance(directory, str):
        directory = Path(directory)
        
    pdf_files = list(directory.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files

def process_directory(
    input_dir: Path, 
    models_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    organized_dirs: Optional[Dict[str, Path]] = None,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 20,
    use_gpu: bool = True,
    batch_size: int = 1,
    save_intermediate: bool = True
) -> List[TextChunk]:
    """Process all PDF files in a directory and return all chunks.
    
    Args:
        input_dir: Directory containing PDF files
        models_dir: Optional path to Docling model artifacts
        output_dir: Optional directory to save intermediate results
        organized_dirs: Optional dictionary with organized subdirectories
        max_chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between chunks
        use_gpu: Whether to use GPU for processing (if available)
        batch_size: Number of files to process before garbage collection
        save_intermediate: Whether to save intermediate results
        
    Returns:
        List of TextChunk objects
    """
    pdf_files = find_pdf_files(input_dir)
    all_chunks = []
    
    # 创建chunks目录（如果需要保存中间结果）
    chunks_dir = None
    if save_intermediate and output_dir:
        chunks_dir = Path(output_dir) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Intermediate chunks will be saved to: {chunks_dir}")
    
    # 批量处理文件
    for i, pdf_path in enumerate(pdf_files):
        try:
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_path}")
            
            # 处理前记录内存使用情况
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage before processing: {mem_before:.2f} MB")
            
            chunks = process_pdf_to_chunks(
                pdf_path, 
                models_dir,
                output_dir,
                organized_dirs,
                max_chunk_size,
                chunk_overlap,
                use_gpu=use_gpu
            )
            
            # 保存中间结果（如果需要）
            if save_intermediate and chunks_dir and chunks:
                chunk_file = chunks_dir / f"{pdf_path.stem}_chunks.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump([chunk.to_dict() for chunk in chunks], f, indent=2)
                logger.info(f"Saved {len(chunks)} chunks to {chunk_file}")
            
            all_chunks.extend(chunks)
            
            # 处理后记录内存使用情况
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage after processing: {mem_after:.2f} MB (change: {mem_after-mem_before:.2f} MB)")
            
            # 在每个文件处理完成后立即进行垃圾回收
            logger.info(f"Running garbage collection after processing {pdf_path}")
            import gc
            gc.collect()
            
            # 记录垃圾回收后的内存使用情况
            mem_after_gc = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage after GC: {mem_after_gc:.2f} MB (freed: {mem_after-mem_after_gc:.2f} MB)")
            
            # 如果达到批处理大小，等待一段时间让系统充分释放资源
            if (i + 1) % batch_size == 0 and i < len(pdf_files) - 1:
                logger.info(f"Processed batch of {batch_size} files, pausing briefly to allow system resource cleanup")
                import time
                time.sleep(2)  # 短暂暂停2秒
            
        except Exception as e:
            logger.error(f"Error processing file {pdf_path}: {e}")
            # 即使处理失败，也进行垃圾回收
            import gc
            gc.collect()
            # 继续处理下一个文件，而不是终止整个过程
            continue
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
