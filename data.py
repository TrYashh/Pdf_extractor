import os
import sys
import re
import csv
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean, group_broken_paragraphs
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class PDFParser:
    """Enhanced PDF Parser with advanced table extraction and text processing"""
    
    def __init__(self, pdf_path: str, output_dir: Optional[str] = None):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir) if output_dir else self.pdf_path.parent
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize sentence transformer model
        self.model = None
        self.chunks = []
        self.tables = []
        self.embeddings = None
        
    def is_likely_tabular_text(self, text: str) -> bool:
        """Enhanced heuristic to detect tabular data in text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
        
        # Check for consistent column delimiters
        delimiter_counts = defaultdict(int)
        for line in lines:
            delimiter_counts['tabs'] += line.count('\t')
            delimiter_counts['pipes'] += line.count('|')
            delimiter_counts['commas'] += line.count(',')
            if re.search(r'\s{3,}', line):  # 3+ spaces for better detection
                delimiter_counts['spaces'] += 1
        
        # If any delimiter appears consistently (>50% of lines)
        if any(count > len(lines) * 0.5 for count in delimiter_counts.values()):
            return True
        
        # Check for numeric alignment patterns
        numeric_columns = 0
        for line in lines:
            parts = re.split(r'\s{3,}', line.strip())
            if len(parts) > 1 and any(re.search(r'\d', part) for part in parts):
                numeric_columns += 1
        
        # Check for header-like patterns
        potential_headers = sum(1 for line in lines[:3] if 
                              any(word.isupper() or word.istitle() for word in line.split()))
        
        return (numeric_columns > len(lines) * 0.6 or 
                potential_headers >= 1 and len(lines) > 3)

    def parse_pdf_with_unstructured(self) -> List[Dict[str, Any]]:
        """Parse PDF document with optimized configuration"""
        try:
            logging.info(f"Processing PDF: {self.pdf_path.name}")
            
            elements = partition_pdf(
                filename=str(self.pdf_path),
                strategy="hi_res",  # Use high-resolution strategy for better table detection
                infer_table_structure=True,
                include_page_breaks=True,
                max_characters=1000000,
                new_after_n_chars=3000,  # Smaller chunks for better processing
                extract_images_in_pdf=False,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=False,
                pdf_extract_line_scale=15,
                pdf_infer_table_structure=True,
                pdf_table_extraction_method="lattice",
                skip_infer_table_types=[],
                languages=["eng"],
                encoding="utf-8",
                ocr_languages="eng",
                ocr_mode="entire_page",
            )
            
            if not elements:
                logging.warning("No content extracted from PDF")
                return []
                
            return self.process_elements(elements)
            
        except Exception as e:
            logging.error(f"PDF parsing failed: {e}")
            # Fallback to basic strategy
            try:
                logging.info("Attempting fallback parsing with basic strategy...")
                elements = partition_pdf(
                    filename=str(self.pdf_path),
                    strategy="fast",
                    infer_table_structure=True,
                )
                return self.process_elements(elements)
            except Exception as fallback_e:
                logging.error(f"Fallback parsing also failed: {fallback_e}")
                return []

    def process_elements(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """Convert extracted elements to structured chunks with enhanced metadata"""
        extracted_chunks = []
        for i, el in enumerate(elements):
            try:
                content = getattr(el, "text", str(el)).strip()
                if not content or len(content) < 3:  # Skip very short content
                    continue

                metadata = {
                    'element_id': i,
                    'element_type': type(el).__name__,
                    'coordinates': getattr(el.metadata, 'coordinates', None),
                    'page_number': getattr(el.metadata, 'page_number', None),
                    'filename': getattr(el.metadata, 'filename', None),
                    'filetype': getattr(el.metadata, 'filetype', None),
                    'parent_id': getattr(el.metadata, 'parent_id', None),
                    'category': getattr(el, 'category', None),
                }
                
                html = getattr(el.metadata, "text_as_html", None)
                
                # Enhanced table detection
                is_table = (html and ("<table" in html or "<td" in html)) or \
                          self.is_likely_tabular_text(content) or \
                          getattr(el, 'category', '') == 'Table'
                
                if is_table:
                    extracted_chunks.append({
                        "type": "table",
                        "content": html if html else content,
                        "raw_content": content,
                        "metadata": metadata
                    })
                else:
                    extracted_chunks.append({
                        "type": "text",
                        "content": content,
                        "metadata": metadata
                    })
                    
            except Exception as e:
                logging.warning(f"Element processing error: {e}")
                continue
                
        logging.info(f"Processed {len(extracted_chunks)} elements from PDF")
        return extracted_chunks

    def detect_delimiter(self, lines: List[str], sample_size: int = 5) -> Optional[str]:
        """Detect the most likely column delimiter in tabular text"""
        delimiters = ['\t', '|', ',', '  ', '   ']  # Added more space variations
        best_delimiter = None
        max_consistency = 0
        
        sample_lines = lines[:min(sample_size, len(lines))]
        
        for delimiter in delimiters:
            counts = []
            valid_splits = 0
            
            for line in sample_lines:
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                counts.append(len(parts))
                if len(parts) > 1:
                    valid_splits += 1
            
            if valid_splits > 0 and len(set(counts)) <= 2:  # Allow some variation
                avg_parts = sum(counts) / len(counts)
                consistency_score = valid_splits * avg_parts
                
                if consistency_score > max_consistency and avg_parts >= 2:
                    max_consistency = consistency_score
                    best_delimiter = delimiter
        
        return best_delimiter

    def smart_split_line(self, line: str) -> List[str]:
        """Intelligently split a line of text into columns"""
        line = line.strip()
        if not line:
            return []
        
        # Try regex patterns for common table formats
        patterns = [
            r'\s*\|\s*',  # Pipe-separated
            r'\s{3,}',    # Multiple spaces
            r'\t+',       # Tabs
        ]
        
        for pattern in patterns:
            parts = [p.strip() for p in re.split(pattern, line) if p.strip()]
            if len(parts) > 1:
                return parts
        
        # Fallback: try to detect column boundaries by looking for word boundaries
        # followed by numeric patterns
        words = line.split()
        if len(words) >= 3:
            # Find potential column breaks
            breaks = [0]
            for i, word in enumerate(words[1:], 1):
                if (re.match(r'^\d+(\.\d+)?%?$', word) or  # Numbers or percentages
                    re.match(r'^\$?\d+(\.\d+)?[KMB]?$', word) or  # Currency/abbreviated numbers
                    (i > 1 and word.isupper() and len(word) <= 5)):  # Short uppercase (codes)
                    breaks.append(i)
            
            if len(breaks) > 1:
                breaks.append(len(words))
                parts = []
                for i in range(len(breaks) - 1):
                    part = ' '.join(words[breaks[i]:breaks[i+1]])
                    if part.strip():
                        parts.append(part.strip())
                if len(parts) > 1:
                    return parts
        
        return [line]

    def fix_table_structure(self, rows: List[List[str]]) -> List[List[str]]:
        """Post-process table rows to fix common structural issues"""
        if not rows:
            return rows
        
        # Remove completely empty rows
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        
        if not rows:
            return rows
        
        # Find the most common row length (mode)
        length_counts = defaultdict(int)
        for row in rows:
            length_counts[len(row)] += 1
        
        if not length_counts:
            return rows
            
        most_common_length = max(length_counts.items(), key=lambda x: x[1])[0]
        
        # Normalize column count
        normalized_rows = []
        for row in rows:
            if len(row) < most_common_length:
                # Pad with empty strings
                row += [''] * (most_common_length - len(row))
            elif len(row) > most_common_length:
                # Try to merge excess columns if they look like continuation
                if most_common_length > 0:
                    merged_row = row[:most_common_length-1]
                    merged_row.append(' '.join(row[most_common_length-1:]))
                    row = merged_row
                else:
                    row = row[:most_common_length]
            normalized_rows.append(row)
        
        rows = normalized_rows
        
        # Detect and handle split headers
        if len(rows) >= 2:
            first_row = rows[0]
            second_row = rows[1]
            
            # Check if first row might be a split header
            first_empty_ratio = sum(1 for cell in first_row if not cell.strip()) / len(first_row)
            if (first_empty_ratio > 0.3 and  # Some empty cells in first row
                any(cell.strip() for cell in first_row) and  # But not all empty
                all(cell.strip() for cell in second_row)):  # Second row has content
                
                merged_header = []
                for first, second in zip(first_row, second_row):
                    combined = f"{first.strip()} {second.strip()}".strip()
                    merged_header.append(combined if combined else second.strip())
                
                rows = [merged_header] + rows[2:]
        
        return rows

    def parse_single_table(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
            """Parse a single HTML or text-based table with improved detection"""
            content = chunk["content"]
            raw_content = chunk.get("raw_content", content)
            rows = []

            try:
                if isinstance(content, str) and "<table" in content.lower():
                    # HTML table parsing
                    soup = BeautifulSoup(content, "html.parser")
                    for tr in soup.find_all("tr"):
                        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
                        if cells:
                            rows.append(cells)
                else:
                    # Text-based table parsing
                    raw_lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
                    lines = []
                    date_regex = re.compile(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+\d{4})\b')

                    for line in raw_lines:
                        matches = list(date_regex.finditer(line))
                        if len(matches) >= 2:
                            for i in range(len(matches)):
                                start = matches[i].start()
                                end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
                                segment = line[start:end].strip()
                                if segment:
                                    lines.append(segment)
                        else:
                            lines.append(line.strip())

                    delimiter = self.detect_delimiter(lines)
                    for line in lines:
                        cells = line.split(delimiter) if delimiter else self.smart_split_line(line)
                        if cells:
                            rows.append([cell.strip() for cell in cells if cell.strip()])

                if rows:
                    rows = self.fix_table_structure(rows)
                    logging.info(f"[DEBUG] Parsed table: {len(rows)} rows × {len(rows[0]) if rows else 0} columns")
                    return {
                        "type": "table",
                        "content": rows,
                        "metadata": chunk.get("metadata", {})
                    }

            except Exception as e:
                logging.warning(f"Table parsing error: {e}")
                cleaned = clean(raw_content, extra_whitespace=True, dashes=True, bullets=True, trailing_punctuation=True)
                if cleaned:
                    return {
                        "type": "text",
                        "content": group_broken_paragraphs(cleaned),
                        "metadata": chunk.get("metadata", {})
                    }

            # fallback
            return {
                "type": "text",
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {})
            }

    
    def parse_table_html(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed_chunks = []
        table_chunks = [chunk for chunk in chunks if chunk["type"] == "table"]
        text_chunks = [chunk for chunk in chunks if chunk["type"] == "text"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(self.parse_single_table, table_chunks)
            parsed_chunks.extend(list(results))

        for chunk in text_chunks:
            try:
                cleaned = clean(
                    chunk["content"],
                    extra_whitespace=True,
                    dashes=True,
                    bullets=True,
                    trailing_punctuation=True
                )
                if cleaned and len(cleaned.strip()) > 5:
                    parsed_chunks.append({
                        "type": "text",
                        "content": group_broken_paragraphs(cleaned),
                        "metadata": chunk.get("metadata", {})
                    })
            except Exception as e:
                logging.warning(f"Text cleaning error: {e}")
                continue
            return parsed_chunks
    

    def clean_and_merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and intelligently merge text chunks"""
        cleaned_chunks = []
        text_buffer = []
        current_page = None
        current_metadata = {}
        
        for chunk in chunks:
            chunk_page = chunk.get("metadata", {}).get("page_number")
            
            if chunk["type"] == "text":
                # Check if we should merge with previous chunks
                should_merge = (
                    len(chunk["content"]) < 200 or  # Small fragments
                    (current_page is not None and chunk_page == current_page) or  # Same page
                    (text_buffer and len(text_buffer[-1]) < 100)  # Previous chunk was small
                )
                
                if should_merge and len(text_buffer) < 5:  # Limit buffer size
                    text_buffer.append(chunk["content"])
                    if not current_metadata:
                        current_metadata = chunk.get("metadata", {})
                    current_page = chunk_page
                else:
                    # Flush buffer if it exists
                    if text_buffer:
                        merged = " ".join(text_buffer)
                        if len(merged.strip()) > 10:  # Skip very short merged content
                            cleaned_chunks.append({
                                "type": "text",
                                "content": group_broken_paragraphs(merged),
                                "metadata": current_metadata
                            })
                        text_buffer = []
                    
                    # Add current chunk
                    if len(chunk["content"].strip()) > 10:
                        cleaned_chunks.append(chunk)
                    current_metadata = chunk.get("metadata", {})
                    current_page = chunk_page
            else:
                # Flush any pending text buffer
                if text_buffer:
                    merged = " ".join(text_buffer)
                    if len(merged.strip()) > 10:
                        cleaned_chunks.append({
                            "type": "text",
                            "content": group_broken_paragraphs(merged),
                            "metadata": current_metadata
                        })
                    text_buffer = []
                    current_metadata = {}
                    current_page = None
                
                # Add non-text chunk
                cleaned_chunks.append(chunk)
        
        # Flush remaining buffer
        if text_buffer:
            merged = " ".join(text_buffer)
            if len(merged.strip()) > 10:
                cleaned_chunks.append({
                    "type": "text",
                    "content": group_broken_paragraphs(merged),
                    "metadata": current_metadata
                })
        
        return cleaned_chunks

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
        """Generate embeddings for text and table content"""
        try:
            if self.model is None:
                logging.info("Loading sentence transformer model...")
                self.model = SentenceTransformer('all-mpnet-base-v2')
            
            flat_chunks = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                if chunk["type"] == "table":
                    # Create searchable text from table
                    table_text_parts = []
                    for row_idx, row in enumerate(chunk["content"]):
                        if row_idx == 0:  # Header row
                            table_text_parts.append("Headers: " + " | ".join(str(cell) for cell in row))
                        else:
                            table_text_parts.append(" | ".join(str(cell) for cell in row))
                    
                    table_text = "\n".join(table_text_parts)
                    flat_chunks.append(table_text)
                    chunk_metadata.append({"chunk_index": i, "type": "table"})
                else:
                    flat_chunks.append(chunk["content"])
                    chunk_metadata.append({"chunk_index": i, "type": "text"})
            
            if flat_chunks:
                embeddings = self.model.encode(flat_chunks, show_progress_bar=True)
                return flat_chunks, embeddings, chunk_metadata
            else:
                return [], np.array([]), []
                
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return [], np.array([]), []

    def export_to_csv(self, tables: List[Dict[str, Any]], output_path: str):
        """Export extracted tables to CSV with improved formatting"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                
                for i, table in enumerate(tables):
                    if i > 0:
                        writer.writerow([])  # Blank line between tables
                    
                    # Add table metadata
                    metadata = table.get("metadata", {})
                    page_num = metadata.get("page_number", "Unknown")
                    writer.writerow([f"=== Table {i+1} (Page: {page_num}) ==="])
                    writer.writerow([])
                    
                    # Write table content
                    for row_idx, row in enumerate(table["content"]):
                        # Clean cells
                        cleaned_row = []
                        for cell in row:
                            cell_str = str(cell).strip()
                            # Remove excessive whitespace
                            cell_str = re.sub(r'\s+', ' ', cell_str)
                            cleaned_row.append(cell_str)
                        writer.writerow(cleaned_row)
            
            logging.info(f"Successfully exported {len(tables)} tables to {output_path}")
            
        except Exception as e:
            logging.error(f"CSV export failed: {e}")

    def export_to_json(self, chunks: List[Dict[str, Any]], output_path: str):
        """Export all processed chunks to JSON"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_chunks = []
            for chunk in chunks:
                serializable_chunk = {
                    "type": chunk["type"],
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {})
                }
                serializable_chunks.append(serializable_chunk)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "pdf_path": str(self.pdf_path),
                    "total_chunks": len(chunks),
                    "tables_count": len([c for c in chunks if c["type"] == "table"]),
                    "text_chunks_count": len([c for c in chunks if c["type"] == "text"]),
                    "chunks": serializable_chunks
                }, f, indent=2, ensure_ascii=False)
                
            logging.info(f"Successfully exported data to {output_path}")
            
        except Exception as e:
            logging.error(f"JSON export failed: {e}")

    def print_summary(self, chunks: List[Dict[str, Any]], tables: List[Dict[str, Any]]):
        """Print a detailed summary of the extraction"""
        print("\n" + "="*60)
        print("PDF PROCESSING SUMMARY")
        print("="*60)
        print(f"PDF File: {self.pdf_path.name}")
        print(f"Total Chunks Extracted: {len(chunks)}")
        print(f"Tables Found: {len(tables)}")
        print(f"Text Chunks: {len(chunks) - len(tables)}")
        
        if tables:
            print("\nTable Details:")
            for i, table in enumerate(tables, 1):
                rows, cols = len(table["content"]), len(table["content"][0]) if table["content"] else 0
                page = table.get("metadata", {}).get("page_number", "Unknown")
                print(f"  Table {i}: {rows} rows × {cols} columns (Page {page})")
                
                # Show first few rows as preview
                if table["content"]:
                    print("    Preview:")
                    preview_rows = table["content"][:3]
                    for row in preview_rows:
                        truncated_row = [str(cell)[:30] + "..." if len(str(cell)) > 30 else str(cell) 
                                       for cell in row]
                        print(f"      {' | '.join(truncated_row)}")
                    if len(table["content"]) > 3:
                        print(f"      ... and {len(table['content']) - 3} more rows")
                print()
        
        # Show page distribution
        page_counts = defaultdict(int)
        for chunk in chunks:
            page = chunk.get("metadata", {}).get("page_number", "Unknown")
            page_counts[page] += 1
        
        if page_counts:
            print("Content Distribution by Page:")
            for page in sorted(page_counts.keys(), key=lambda x: x if x != "Unknown" else float('inf')):
                print(f"  Page {page}: {page_counts[page]} chunks")
        
        print("="*60)

    def process(self, export_csv: bool = True, export_json: bool = True) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # 1. Parse PDF
            logging.info("Step 1: Parsing PDF...")
            raw_chunks = self.parse_pdf_with_unstructured()
            if not raw_chunks:
                logging.error("No content extracted from PDF")
                return {"success": False, "error": "No content extracted"}
            
            # 2. Clean and merge chunks
            logging.info("Step 2: Cleaning and merging chunks...")
            cleaned_chunks = self.clean_and_merge_chunks(raw_chunks)
            
            # 3. Parse tables
            logging.info("Step 3: Parsing tables...")
            parsed_chunks = self.parse_table_html(cleaned_chunks)
            
            # 4. Generate embeddings
            logging.info("Step 4: Generating embeddings...")
            flat_chunks, embeddings, chunk_metadata = self.embed_chunks(parsed_chunks)
            
            # Store results
            self.chunks = parsed_chunks
            self.tables = [chunk for chunk in parsed_chunks if chunk["type"] == "table"]
            self.embeddings = embeddings

            logging.info(f"Tables detected: {len(self.tables)}")  # <--- Add this line

            # 5. Export results
            base_name = self.pdf_path.stem
            
            if export_csv:
                if self.tables:
                    csv_path = self.output_dir / f"{base_name}_tables.csv"
                    logging.info(f"Exporting tables to CSV: {csv_path}")  # <--- Add this line
                    self.export_to_csv(self.tables, str(csv_path))
                else:
                    logging.warning("No tables found, skipping CSV export.")  # <--- Add this line
            
            if export_json:
                json_path = self.output_dir / f"{base_name}_data.json"
                self.export_to_json(parsed_chunks, str(json_path))
            
            # 6. Print summary
            self.print_summary(parsed_chunks, self.tables)
            
            return {
                "success": True,
                "chunks": parsed_chunks,
                "tables": self.tables,
                "embeddings": embeddings,
                "flat_chunks": flat_chunks,
                "chunk_metadata": chunk_metadata
            }
            
        except Exception as e:
            logging.error(f"Processing failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf> [output_directory]")
        print("\nExample:")
        print("  python pdf_parser.py document.pdf")
        print("  python pdf_parser.py document.pdf ./output/")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize parser
    parser = PDFParser(pdf_path, output_dir)
    
    # Process PDF
    result = parser.process()
    
    if result["success"]:
        logging.info("Processing completed successfully!")
        sys.exit(0)
    else:
        logging.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()