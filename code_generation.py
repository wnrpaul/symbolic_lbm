# code_generation.py
import sympy as sp

def generate_cpp_code(expression, symbols):
    # Utilisez sp.ccode en remplaçant les symboles par ceux du dictionnaire
    code = sp.ccode(expression)
    for logical_name, symbol_name in symbols.items():
        code = code.replace(str(logical_name), symbol_name)
    return code


def generate_latex_code(expression, symbols):
    # Utilisez sp.latex en remplaçant les symboles par ceux du dictionnaire
    code = sp.latex(expression)
    for logical_name, symbol_name in symbols.items():
        code = code.replace(str(logical_name), symbol_name)
    return code
