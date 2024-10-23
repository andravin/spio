#!/bin/bash


for file in *.ubj; do
  new_name=$(echo "$file" | sed 's/devicemodel__.*__sm/archmodel__sm/')
  cp "$file" "$new_name"
done
