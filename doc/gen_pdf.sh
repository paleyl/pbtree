#!/bin/sh
# run this script to convert markdown to pdf
pandoc -f markdown pbtree.md -o pbtree.pdf --pdf-engine=xelatex -V CJKmainfont="Songti SC"