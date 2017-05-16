#!/bin/bash
echo searching string \"$1\" in current folder
find . -name "*.cc" -o -name "*.h" -o -name "*.c" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.l" -o -name "*.y" | xargs grep -n "$1"
