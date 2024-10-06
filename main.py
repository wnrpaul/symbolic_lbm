# main.py
import os
import argparse
from equilibrium_functions import GuoImproved, GradHermite
from symbols_mapping import DEFAULT_SYMBOLS, LATEX_SYMBOLS, load_custom_symbols

def main():
    parser = argparse.ArgumentParser(description='Calcul des fonctions d\'équilibre LBM.')
    parser.add_argument('--D', type=int, default=3, help='Dimension spatiale.')
    parser.add_argument('--Q', type=int, default=19, help='Nombre de vitesses du réseau.')
    parser.add_argument('--eq_type', type=str, default='Guo-Improved', help='Type de fonction d\'équilibre.')
    parser.add_argument('--is_thermal', action='store_true', help='Activer le mode thermique.')
    parser.add_argument('--order_0', type=int, default=4, help='Ordre de l\'expansion Hermite pour l\'équilibre.')
    parser.add_argument('--output_format', type=str, default='cpp', help='Format de sortie (cpp, latex, custom).')
    parser.add_argument('--user_equilibrium_path', type=str, help='Chemin vers le script de fonction d\'équilibre utilisateur.')
    parser.add_argument('--user_symbol_path', type=str, help='Chemin vers le json de mapping des symboles utilisateur.')

    args = parser.parse_args()

    if args.output_format == 'cpp':
        symbols = DEFAULT_SYMBOLS
    elif args.output_format == 'latex':
        symbols = LATEX_SYMBOLS
    elif args.output_format == 'custom':
        if not args.user_symbol_path:
            raise ValueError("Le chemin vers le fichier de mapping des symboles doit être fourni.")
        # L'utilisateur peut fournir son propre mapping de symboles
        symbols = load_custom_symbols(args.user_symbol_path)
    else:
        raise ValueError(f"Format de sortie inconnu : {args.output_format}")

    if args.eq_type == 'Grad-Hermite':
        eq_class = GradHermite(args.D, args.Q, args.is_thermal, args.order_0, symbols)
    # if args.eq_type == 'Guo-Improved':
    #     eq_func = GuoImproved(args.D, args.Q, args.is_thermal, args.order_0)
    # elif args.eq_type == 'user_defined':
    #     if args.user_equilibrium_path:
    #         UserEquilibriumClass = load_user_equilibrium(args.user_equilibrium_path)
    #         eq_func = UserEquilibriumClass(args.D, args.Q, args.is_thermal, args.order_0)
    #     else:
    #         raise ValueError("Le chemin vers le script de fonction d'équilibre utilisateur doit être fourni.")
    else:
        raise ValueError(f"Type de fonction d'équilibre inconnu : {args.eq_type}")

    # D = 3
    # Q = 19
    # is_thermal = True
    # order_0 = 4

    # Initialisation de la fonction d'équilibre
    # eq_func = GuoImproved(D, Q, is_thermal, order_0, symbols)
    print(eq_class.compute())

if __name__ == '__main__':
    main()
