#!/bin/sh

find . | grep -v thirdparty | grep -v build | grep -v \./data | \
grep -v git | grep -E "(\.cxx|\.h|\.txt|\.proto|\.sh|\.py)" | grep -v bak |\
while read line; do cat $line;done| wc -l
