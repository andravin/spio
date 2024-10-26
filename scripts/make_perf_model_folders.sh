#!/bin/bash

declare -A model_dirs

# Process devicemodel files
for file in devicemodel__*__*.ubj; do
  device_name=$(echo "$file" | sed -n 's/devicemodel__\(.*\)__\(.*\)__spio_.*\.ubj/\1/p')
  arch_name=$(echo "$file" | sed -n 's/devicemodel__\(.*\)__\(.*\)__spio_.*\.ubj/\2/p')
  dir_name="devicemodel__${device_name}__${arch_name}"
  mkdir -p "$dir_name"
  mv "$file" "$dir_name/"
  model_dirs["$dir_name"]=1
done

# Process archmodel files
for file in archmodel__*__*.ubj; do
  arch_name=$(echo "$file" | sed -n 's/archmodel__\(.*\)__spio_.*\.ubj/\1/p')
  dir_name="archmodel__${arch_name}"
  mkdir -p "$dir_name"
  mv "$file" "$dir_name/"
  model_dirs["$dir_name"]=1
done

# Create .tgz for each unique directory
for dir_name in "${!model_dirs[@]}"; do
  tar -czf "$dir_name.tgz" "$dir_name"
done