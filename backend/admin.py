from django.contrib import admin
from .models import Uploaded_File

# Enregistrement de votre modèle ici dans l'interface d'administration Django
# Cela permet de gérer les fichiers téléchargés via l'interface admin.
admin.site.register(Uploaded_File)  # Enregistre le modèle Uploaded_File pour le rendre disponible dans l'admin
