#!/bin/bash

# Define the array of items
items=("gemm" "lu" "covariance" "jacobi_2d")

# Clear the screen
clear

# Iterate over each item
for item in "${items[@]}"; do

    # Compile the program
    g++ -I./include MLIR2EGG.cpp -o mlir_parser

    # Run the program with the current item
    ./mlir_parser -i "mlir/${item}/${item}.mlir" -o "mlir_output/${item}.txt" -m "mlir_output/${item}_update.mlir"
done

