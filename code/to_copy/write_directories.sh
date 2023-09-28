#!/bin/bash

# Set the path of the input file
input_file="./dir_names.txt"

# Read each line of the input file and create the directory
while read -r line
do
  mkdir -p "$line"
done < "$input_file"

