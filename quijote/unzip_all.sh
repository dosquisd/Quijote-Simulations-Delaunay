#!/bin/bash

# Exit on error
set -e

echo "Starting to unzip all ZIP files in current directory..."

# Find all zip files (case insensitive)
find . -maxdepth 1 -type f \( -iname "*.zip" \) -print0 | while read -d $'\0' file; do
    # Extract filename for display
    filename=$(basename "$file")
    
    echo "Processing: $filename"
    
    if unzip -t "$file" >/dev/null 2>&1; then
        echo "Unzipping $filename..."
        unzip -o "$file"
        echo "Successfully unzipped $filename"
        echo "----------------------------------------"
    else
        echo "Error: $filename appears to be corrupted or is not a valid ZIP file"
        echo "----------------------------------------"
        continue
    fi
done

echo "Finished processing all ZIP files"

