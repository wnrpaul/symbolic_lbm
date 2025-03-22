# Symbolic Lattice Boltzmann Method (LBM) Calculation

This repository contains a Python-based framework for performing symbolic Lattice Boltzmann Method (LBM) calculations. It supports various equilibrium functions and output formats, including C++, LaTeX, and custom formats. The framework is designed to be extensible and configurable, allowing users to define custom equilibrium functions and symbols.

## Features

- **Equilibrium Functions**: Supports multiple equilibrium functions, including Grad-Hermite, Guo-Improved, Unified, etc...
- **Output Formats**: Generates output in C++, LaTeX, and custom formats.
- **Optimization**: Optimizes the number of floating-point operations (FLOPs) in the generated code.
- **Configurable Logging**: Allows configuring the logging level to control the verbosity of the output.

## Installation

To use this framework, you must install Python on your machine. You can install the required dependencies using pip:

```sh
pip install sympy numpy
```

## Usage

The main entry point for the framework is the `main.py` script. You can run the script with various command-line arguments to configure the calculation.

```sh
python main.py --D 3 --Q 19 --eq_type grad-hermite --output_format cpp --log_level INFO
```

### Command-Line Arguments

- `--D`: Spatial dimension (default: 3).
- `--Q`: Number of lattice velocities (default: 19).
- `--eq_type`: Type of equilibrium function (default: 'grad-hermite').
- `--is_thermal`: Enable thermal equilibrium.
- `--order_0`: Order of Hermite expansion for the equilibrium (default: 3).
- `--output_format`: Output format (cpp, latex, custom) (default: 'cpp').
- `--optim_flops`: Optimize the number of FLOPs.
- `--user_equilibrium_path`: Path to user equilibrium function script.
- `--user_symbol_path`: Path to user symbols mapping in JSON.
- `--log_level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: 'WARNING').

## Modules

### `code_generation.py`

It contains functions for generating and writing C++ code from symbolic matrices.

- `write_output_file(content, filename)`: Writes the content to the specified file, ensuring the directory exists.
- `generate_cpp_code(matrix, place_holder, order)`: Generates C++ code from a symbolic matrix.
- `write_cpp_matrix_from_cse(cse_matrix_tuple, place_holder, order, filename)`: Generates C++ code from an optimized matrix with Common Sub-Expressions (CSE) and writes it to a file.
- `optimize_matrix_flops(matrix, sub_list)`: Optimizes the number of FLOPs in a matrix using SymPy.

### `equilibrium_functions.py`

Contains the definitions of equilibrium functions for the lattice Boltzmann method (LBM).

- `EquilibriumFunction`: Base class for equilibrium functions.
- `GradHermite`: Grad-Hermite equilibrium function.
- `D3Q19GuoImproved`: Guo-Improved equilibrium function.
- `D3Q19Unified`: Unified equilibrium function.
- `D3Q19Iso`: Isotropic equilibrium function.

### `symbols_mapping.py`

Contains mappings for default and custom symbols used in the calculations.

- `DEFAULT_SYMBOLS`: Default symbols mapping.
- `LATEX_SYMBOLS`: LaTeX symbols mapping.
- `load_custom_symbols(path)`: Loads custom symbols mapping from a JSON file.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
