# main.py
import os
import argparse
import sympy as sp
import logging
from code_generation import write_output_file, generate_cpp_code, optimize_matrix_flops, write_cpp_matrix_from_cse
from equilibrium_functions import D3Q19Iso, D3Q19Unified, D3Q19GuoImproved, GradHermite
from symbols_mapping import DEFAULT_SYMBOLS, LATEX_SYMBOLS, load_custom_symbols


def main():
    parser = argparse.ArgumentParser(description='Symbolic LBM calculation.')
    parser.add_argument('--D', type=int, default=3, 
                        help='Spatial dimension.')
    parser.add_argument('--Q', type=int, default=19,
                        help='Number of lattice velocities.')
    parser.add_argument('--eq_type', type=str, default='grad-hermite',
                        help='Type of equilibrium function.')
    parser.add_argument('--is_thermal', action='store_true',
                        help='Enable thermal equilibrium.')
    parser.add_argument('--order_0', type=int, default=3,
                        help='Order of Hermite expansion for the equilibrium.')
    parser.add_argument('--output_format', type=str, default='cpp',
                        help='Output format (cpp, latex, custom).')
    parser.add_argument('--optim_flops', action='store_true', 
                        help='Optimize the number of FLOPs.')
    parser.add_argument('--user_equilibrium_path', type=str,
                        help='Path to user equilibrium function script.')
    parser.add_argument('--user_symbol_path', type=str,
                        help='Path to user symbols mapping in JSON.')
    # Add arguments for logging level
    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    parser.add_argument('--log_level', type=str, choices=choices,
                        default='WARNING', 
                        help=f'Set the logging level: {choices}.')

    args = parser.parse_args()

    # Configure logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid logging level: {args.log_level}')
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.output_format == 'cpp':
        symbols = DEFAULT_SYMBOLS
        logging.info("Using C++ symbols.")
    elif args.output_format == 'latex':
        symbols = LATEX_SYMBOLS
        logging.info("Using LaTeX symbols.")
    elif args.output_format == 'custom':
        if not args.user_symbol_path:
            raise ValueError(
                "Path to user symbols mapping file must be provided.")
        # User can provide their own symbols mapping
        logging.info(f"Use custom symbols from {args.user_symbol_path}")
        symbols = load_custom_symbols(args.user_symbol_path)
    else:
        raise ValueError(f"Unknown output format: {args.output_format}")

    if args.eq_type == 'grad-hermite':
        eq_class = GradHermite(
            args.D, args.Q, args.is_thermal, args.order_0, symbols)
    elif args.eq_type == 'guo-improved':
        eq_class = D3Q19GuoImproved(symbols=symbols)
    elif args.eq_type == 'unified':
        eq_class = D3Q19Unified(symbols=symbols)
    elif args.eq_type == 'iso-v1':
        eq_class = D3Q19Iso(version=1, symbols=symbols)
    elif args.eq_type == 'iso-v2':
        eq_class = D3Q19Iso(version=2, symbols=symbols)
    # elif args.eq_type == 'custom':
        # if args.user_equilibrium_path:
        #     UserEquilibriumClass = load_user_equilibrium(args.user_equilibrium_path)
        #     eq_func = UserEquilibriumClass(args.D, args.Q, args.is_thermal, args.order_0)
        # else:
        #     raise ValueError("Path to user equilibrium function script must be provided.")
    else:
        raise ValueError(f"Unknown equilibrium function type: {args.eq_type}")

    # Equilibrium function initialization
    f_eq = eq_class.compute_feq()

    # Writing of the equilibrium function
    if args.output_format == 'cpp':
        filename = os.path.join('output', args.output_format, eq_class.name + '.cpp')
        logging.info(f"Generating C++ code for {eq_class.name}: {filename}")
        content = generate_cpp_code(matrix=f_eq, 
                         place_holder=sp.MatrixSymbol(symbols['fEq'], eq_class.Q, 1), 
                         order=eq_class.order_0)
        write_output_file(content, filename=filename)
    
    if args.optim_flops:
        logging.info("Optimizing the number of FLOPs:")
        cse_matrix_tuple = optimize_matrix_flops(f_eq, sub_list=[])
        if args.output_format == 'cpp':
            filename = os.path.join('output', args.output_format, eq_class.name + '_optim.cpp')
            write_cpp_matrix_from_cse(cse_matrix_tuple=cse_matrix_tuple,
                                  place_holder=sp.MatrixSymbol(symbols['fEq'], eq_class.Q, 1),
                                  order=eq_class.order_0,
                                  filename=filename)

    
    return eq_class, f_eq


if __name__ == '__main__':
    fClass, fEq = main()
