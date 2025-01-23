#!/usr/bin/env python
"""Utilitaire en ligne de commande de Django pour les tâches administratives."""
import os
import sys

def main():
    """Exécute les tâches administratives."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'M1_S_F.settings')  # Définit la configuration par défaut du projet Django
    try:
        from django.core.management import execute_from_command_line  # Importe la commande d'exécution de Django
    except ImportError as exc:
        # Exception levée si Django n'est pas correctement installé
        raise ImportError(
            "Impossible d'importer Django. Assurez-vous qu'il est installé et "
            "qu'il est disponible dans votre variable d'environnement PYTHONPATH. "
            "Avez-vous oublié d'activer un environnement virtuel ?"
        ) from exc
    execute_from_command_line(sys.argv)  # Exécute la commande passée en ligne de commande

if __name__ == '__main__':
    main()  # Point d'entrée du script
