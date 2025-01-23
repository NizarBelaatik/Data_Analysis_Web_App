from django.db import models

# Modèle pour les fichiers téléchargés
class Uploaded_File(models.Model):
    file_id = models.CharField(max_length=255, blank=True)  # ID unique du fichier
    file_name = models.CharField(max_length=255, blank=True)  # Nom du fichier
    file_format = models.CharField(max_length=255, blank=True)  # Format du fichier
    file = models.FileField(upload_to='uploads', blank=True)  # Chemin du fichier téléchargé
    file_date = models.DateTimeField(auto_now=True, null=True, blank=True)  # Date de modification
