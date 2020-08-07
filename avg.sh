#!/bin/bash

if [ $# -gt 0 ]; then
    pat=$1
else
    pat="gather"
fi

echo PATTERN=$pat

for f in *.txt; do
    echo -n "$f "
    grep "$pat" "$f" | \
        awk -F: '{ total += $2; count++ } END { print total / count }'
done
