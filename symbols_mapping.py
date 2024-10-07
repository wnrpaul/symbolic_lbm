import json

DEFAULT_SYMBOLS = {
    'cs': 'cs',
    'rho': 'rho',           
    'tp': 'tp',
    'ux': 'ux',
    'uy': 'uy',
    'uz': 'uz',
    'fx': 'fx',
    'fy': 'fy',
    'fz': 'fz',
    'ups': 'ups',
    'fEq': 'fEq',
    # Ajoutez d'autres symboles si nécessaire
}

# TEST_SYMBOLS = {
#     rho: sp.symbols('a'),           
#     tp: sp.symbols('b'),
#     ux: sp.symbols('c'),
#     uy: sp.symbols('d'),
#     uz: sp.symbols('e'),
# }

LATEX_SYMBOLS = {
    'cs': r'c_s',
    'rho': r'\rho',
    'tp': r'\theta',
    'ux': r'u_x',
    'uy': r'u_y',
    'uz': r'u_z',
    'fx': r'g_x',
    'fy': r'g_y',
    'fz': r'g_z',
    'ups': r'\Upsilon',
    # Ajoutez d'autres symboles si nécessaire
}


def load_custom_symbols(path):
    with open(path, 'r') as f:
        custom_symbols = json.load(f)
    return custom_symbols

