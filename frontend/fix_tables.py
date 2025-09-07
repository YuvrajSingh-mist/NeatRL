#!/usr/bin/env python3
"""
Normalize and fix Markdown tables by:
- Replacing unicode dashes in separator rows
- Ensuring separator uses --- per column
- Merging wrapped table rows that spill across multiple lines

Usage:
  python3 fix_tables.py _smolhub/playground-05-storyllama.md _smolhub/playground-06-storymixtral.md _models/37-whisper.md
  or give a directory to process all .md files within it
"""

import sys
import os
from pathlib import Path

def is_table_separator(line: str) -> bool:
    if '|' not in line:
        return False
    # replace unicode dashes
    line = line.replace('—', '-').replace('–', '-')
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 2:
        return False
    # a separator line should consist only of dashes and optional colons per cell
    for cell in parts:
        cell_no_space = cell.replace('-', '').replace(':', '')
        if cell and cell_no_space != '':
            return False
    return True

def normalize_separator(line: str, num_cols: int) -> str:
    line = line.replace('—', '-').replace('–', '-')
    # Build a consistent separator with alignment preserved if present
    parts = line.split('|')
    normalized = []
    for idx, raw in enumerate(parts):
        cell = raw.strip()
        if not cell:
            normalized.append('')
            continue
        left = ':' if cell.startswith(':') else ''
        right = ':' if cell.endswith(':') else ''
        normalized.append(f"{left}{'-'*3}{right}")
    out = '|'.join(normalized)
    # Ensure we have at least num_cols cells (including content cells, excluding empties)
    # Add missing chunks if needed
    content_cells = [c for c in normalized if c != '']
    if len(content_cells) < num_cols:
        extras = '|' + '|'.join(['---']*(num_cols - len(content_cells)))
        out = out + extras
    return out

def count_cells(line: str) -> int:
    # count non-empty cells
    return len([c for c in line.split('|') if c.strip() != ''])

def fix_tables_in_text(text: str) -> str:
    lines = text.split('\n')
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of a table
        if '|' in line and i + 1 < len(lines) and '|' in lines[i+1]:
            header = line
            sep = lines[i+1]
            if is_table_separator(sep):
                # Start table processing
                fixed.append(header)
                header_cols = count_cells(header)
                fixed.append(normalize_separator(sep, header_cols))
                i += 2
                # Process rows until a non-table line is hit
                while i < len(lines):
                    row_line = lines[i]
                    if '|' not in row_line:
                        break
                    # Merge wrapped rows: accumulate until enough cells
                    merged = row_line.strip()
                    # If merged contains less than header_cols cells, pull next lines
                    while count_cells(merged) < header_cols and i + 1 < len(lines) and '|' in lines[i+1]:
                        i += 1
                        merged = (merged.rstrip() + ' ' + lines[i].strip()).strip()
                    # Replace unicode dashes in data cells that might break formatting
                    merged = merged.replace('—', '-').replace('–', '-')
                    fixed.append(merged)
                    i += 1
                # Ensure blank line after table for proper rendering
                if i < len(lines) and lines[i].strip() != '':
                    fixed.append('')
                continue
        # default path
        fixed.append(line)
        i += 1
    return '\n'.join(fixed)

def process_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding='utf-8')
        fixed = fix_tables_in_text(original)
        if fixed != original:
            path.write_text(fixed, encoding='utf-8')
            print(f"✅ Fixed tables: {path}")
            return True
        else:
            print(f"ℹ️ No table changes needed: {path}")
            return False
    except Exception as e:
        print(f"❌ Failed to process {path}: {e}")
        return False

def main(argv):
    if len(argv) < 2:
        print("Usage: python3 fix_tables.py <file_or_dir> [more files/dirs...]")
        return 1
    changed = 0
    for arg in argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for md in p.rglob('*.md'):
                if process_file(md):
                    changed += 1
        elif p.is_file():
            if process_file(p):
                changed += 1
        else:
            print(f"Skipping non-existent path: {arg}")
    print(f"\n📊 Table fixing complete. Files changed: {changed}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))




