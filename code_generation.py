# code_generation.py
import sympy as sp
import os
import logging
from sympy.codegen.rewriting import create_expand_pow_optimization


def write_output_file(content, filename):
    """
    Writes the content to the specified file, ensuring the directory exists.
    """
    path = os.path.dirname(filename)
    os.makedirs(path, exist_ok=True)
    with open(filename, 'w') as f:
        f.write(content)
    logging.info(f"Successfully wrote output file in '{filename}'.")


def generate_cpp_code(matrix, place_holder, order):
    """
    Writes the matrix in C++ code format.

    Args:
        matrix: SymPy matrix to write.
        place_holder (MatrixSymbol): Symbol of the matrix in the generated code.
        order (int): Order of the expansion optimization.
        filename (str): Output file name.
    
    Returns:
        str: The content of the generated code.
    """
    expand_opt = sp.codegen.rewriting.create_expand_pow_optimization(order)
    content = sp.ccode(expand_opt(matrix),
                       assign_to=place_holder, standard='c99')
    return content


def write_cpp_matrix_from_cse(cse_matrix_tuple, place_holder, order, filename):
    """
    Generates C++ code from an optimized matrix with Common-Sub-Expressions (CSE) and writes it to a file.

    Args:
        cse_matrix_tuple (tuple): Tuple containing the common subexpressions and the optimized matrix.
        place_holder (MatrixSymbol): Symbol representing the matrix in the generated code.
        order (int): Order for the power expansion optimization.
        filename (str): Output file name.
    """
    # Create the expansion function based on the order
    expand_opt = create_expand_pow_optimization(order)

    # Ensure the output directory exists
    path = os.path.dirname(filename)
    os.makedirs(path, exist_ok=True)

    try:
        with open(filename, 'w') as ffile:
            # Write the common subexpressions
            for var, expr in cse_matrix_tuple[0]:
                expr_expanded = sp.expand(expr)
                code = sp.ccode(expr_expanded, assign_to=var, standard='c99')
                ffile.write('const double ' + code + ';\n')
            # Write the final matrix using the subexpressions
            # [0] if cse_matrix_tuple_opt[1] is a list
            expr_final = expand_opt(cse_matrix_tuple[1])
            code_final = sp.ccode(
                expr_final, assign_to=place_holder, standard='c99')
            ffile.write(code_final)
        logging.info(f"Successfully generated C++ code in '{filename}'.")
    except Exception as e:
        logging.error(f"Error during C++ code generation: {e}")
        raises


def optimize_matrix_flops(matrix, sub_list):
    """
    Optimizes the number of FLOPs in a matrix using SymPy.

    This function simplifies the matrix, finds common subexpressions (CSE),
    and applies substitutions to optimize the calculations.

    Args:
        matrix (Matrix): SymPy matrix to optimize in terms of FLOPs.
        sub_list (list): List of substitution dictionaries to apply.

    Returns:
        tuple: A tuple containing the common subexpressions and the optimized matrix.
    """
    # Count the total number of operations, when visual arg is used, it shows
    # all the types of operations
    count_op = sp.count_ops(matrix)
    logging.info(
        f'{f"Number of operations:":<35}' +
        f'{str(sp.count_ops(matrix, visual=True)):<60}' +
        f'{f"FLOPs = {count_op}":<10}')

    # Simplification of each coefficient of the matrix
    var_simplify = sp.Matrix([sp.simplify(aij) for aij in matrix])
    count_op_simplify = sp.count_ops(var_simplify)
    logging.info(
        f'{"After simplification:":<35}' +
        f'{str(sp.count_ops(var_simplify, visual=True)):<60}' +
        f'{f"FLOPs = {count_op_simplify}":<10}')

    # Compute the Common SubExpression (cse) and the associated matrix
    cse_mat_tup = sp.cse(var_simplify)
    count_op_cse = sp.count_ops(cse_mat_tup)
    logging.info(
        f'{"After common subexp. generation:":<35}' +
        f'{str(sp.count_ops(cse_mat_tup, visual=True)):<60}' +
        f'{f"FLOPs = {count_op_cse}":<10}')
    cse_list_of_tuple = cse_mat_tup[0]
    cse_matrix = cse_mat_tup[1][0]

    # Substitute all the expressions using sub_list
    for sub in sub_list:
        cse_list_of_tuple = [(cse_list_of_tuple[i][0],
                              cse_list_of_tuple[i][1].subs(sub))
                             for i in range(len(cse_list_of_tuple))]
        cse_matrix = cse_matrix.subs(sub)
    cse_mat_tup_opt = (cse_list_of_tuple, cse_matrix)
    count_op_sub = sp.count_ops(cse_mat_tup_opt)
    logging.info(
        f'{"After replacing subexpression:":<35}' +
        f'{str(sp.count_ops(cse_mat_tup_opt, visual=True)):<60}' +
        f'{f"FLOPs = {count_op_sub}":<10}')
    logging.info(f'FLOPs reduction: {100*(1-count_op_sub/count_op):.2f} % ')
    return cse_mat_tup_opt
