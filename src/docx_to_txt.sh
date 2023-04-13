#!/bin/bash

for file in *; 
do 
    pandoc -f docx -s "$file" -o "$file.txt"
done

for file in *; 
do 
    mv "${file}" "${file/.txt.docx}"
done
