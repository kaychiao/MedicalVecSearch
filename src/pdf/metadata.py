#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

@dataclass
class NCCNMetadata:
    """Metadata for NCCN guideline documents."""
    filename: str
    cancer_name: str
    year: Optional[str] = None
    version: Optional[str] = None
    language: str = "English"
    author: str = "NCCN"
    source_path: Optional[str] = None

def extract_metadata_from_filename(filepath: Path) -> NCCNMetadata:
    """Extract metadata from medical document filename.
    
    Supports multiple formats:
    1. CancerName_Year.Version_Language.pdf (e.g., Non-SmallCellLungCancer_2025.V3_EN.pdf)
    2. NCCN_CancerName_Year_Version.pdf (e.g., NCCN_Breast_2023_v2.pdf)
    """
    filename = filepath.name
    stem = filepath.stem  # Filename without extension
    
    # Default values
    cancer_name = "Unknown"
    year = None
    version = None
    language = "English"
    
    # Try to extract metadata based on the new format: CancerName_Year.Version_Language.pdf
    parts = stem.split('_')
    
    if len(parts) >= 2:
        # Format: CancerName_Year.Version_Language.pdf
        if '.' in parts[1]:
            # First part is cancer name
            cancer_name = parts[0]
            
            # Second part contains year and version (e.g., 2025.V3)
            year_version = parts[1].split('.')
            if len(year_version) >= 2:
                year = year_version[0]
                version = year_version[1]
            
            # Third part is language (if exists)
            if len(parts) >= 3:
                language = parts[2]
        
        # Legacy format: NCCN_CancerName_Year_Version.pdf
        elif len(parts) >= 2 and parts[0].upper() == "NCCN":
            cancer_name = parts[1]
            
            # Extract year (4 digit number)
            year_match = re.search(r'20\d{2}', stem)
            if year_match:
                year = year_match.group(0)
            
            # Extract version (v followed by number)
            version_match = re.search(r'v\d+(\.\d+)?', stem)
            if version_match:
                version = version_match.group(0)
    
    # Log the extracted metadata
    logger.info(f"Extracted metadata from {filename}: cancer_name={cancer_name}, year={year}, version={version}, language={language}")
    
    return NCCNMetadata(
        filename=filename,
        cancer_name=cancer_name,
        year=year,
        version=version,
        language=language,
        source_path=str(filepath)
    )
