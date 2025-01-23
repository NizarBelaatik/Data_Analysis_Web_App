from django import template

# Création d'une bibliothèque de balises pour les filtres personnalisés
register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Filtre personnalisé pour accéder aux valeurs d'un dictionnaire par clé"""
    return dictionary.get(key)
