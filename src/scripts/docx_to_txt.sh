#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: docx_to_txt.sh <directory>"
    exit
fi

DIR="$(realpath $1)"

for file in $DIR/*.docx; do
    if [[ -f $file ]]; then
#        echo "Converting $file to ${file/.txt.docx}.txt"
        pandoc -s --from="docx" --to="plain" --wrap=none --output="${file/.txt.docx}.txt" "$file"
        rm $file
    fi
done
