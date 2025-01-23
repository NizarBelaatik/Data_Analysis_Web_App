from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.models import User
from django.http import JsonResponse, Http404

# Importation du modèle de fichiers téléchargés
from .models import Uploaded_File

# Importation des bibliothèques nécessaires pour le traitement des données
import pandas as pd
import os
import random
import string
import matplotlib
matplotlib.use('Agg')  # Utilisation d'un backend non interactif pour Matplotlib
import matplotlib.pyplot as plt
 
import seaborn as sns
from django.conf import settings

# Importation des bibliothèques pour l'apprentissage automatique
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Importation des outils pour manipuler les fichiers et les flux
from tempfile import NamedTemporaryFile
from asgiref.sync import sync_to_async
from io import BytesIO
import base64

# Vue pour afficher la page d'accueil et la liste des fichiers
# Affiche tous les fichiers téléchargés
def Home(request):
    files_list = Uploaded_File.objects.all()  # Récupère tous les fichiers depuis la base de données
    return render(request, 'html/main.html', {'files_list': files_list})  # Rend la page avec la liste des fichiers

# Fonction pour générer un ID unique pour chaque fichier téléchargé
def generate_unique_file_id():
    while True:
        # Génère un ID aléatoire en combinant des lettres majuscules et des chiffres
        file_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        if not Uploaded_File.objects.filter(file_id=file_id).exists():
            return file_id

# Vue pour télécharger un fichier
# Enregistre le fichier dans la base de données et sur le système de fichiers
def Upload_File(request):
    if request.method == "POST":
        upFile = request.FILES.get('file')  # Récupère le fichier depuis la requête

        if upFile:
            file_name = upFile.name  # Récupère le nom du fichier
            file_format = file_name.split('.')[-1]  # Récupère l'extension du fichier

            # Génère un ID unique pour le fichier
            fileId = generate_unique_file_id()

            # Enregistre le fichier dans la base de données
            Uploaded_File.objects.create(
                file_id=fileId,
                file_name=file_name,
                file_format=file_format,
                file=upFile
            )

            return JsonResponse({'status': 'succès'})  # Retourne un message de succès

    return JsonResponse({'status': 'échec'})  # Retourne un message d'échec si aucun fichier n'est téléchargé

# Vue pour supprimer un fichier
def remove_file(request, file_id):
    try:
        # Récupère le fichier correspondant à l'ID fourni
        file_to_remove = Uploaded_File.objects.get(file_id=file_id)

        # Supprime le fichier du système de fichiers si il existe
        if file_to_remove.file:
            file_to_remove.file.delete(save=False)

        # Supprime l'enregistrement du fichier dans la base de données
        file_to_remove.delete()

        # Redirige vers la page d'accueil après suppression
        return redirect('home')

    except Uploaded_File.DoesNotExist:
        # Retourne une erreur 404 si le fichier n'existe pas
        raise Http404("Fichier introuvable")

# Vue principale pour afficher les données d'un fichier spécifique
def MAIN(request, file_id):
    data = []
    uploaded_file = Uploaded_File.objects.get(file_id=file_id)  # Récupère le fichier correspondant
    file_path = uploaded_file.file.path  # Obtient le chemin du fichier

    # Lit le fichier CSV avec Pandas
    df = pd.read_csv(file_path, encoding='latin1')
    df_headers = df.columns.tolist()  # Récupère les en-têtes des colonnes
    df_clean = df.loc[~df.isnull().any(axis=1)]  # Nettoie les lignes contenant des valeurs nulles
    data = df_clean.to_dict(orient='records')  # Convertit les données en une liste de dictionnaires

    # Retourne la page avec les données et les en-têtes des colonnes
    return render(request, 'html/chart.html', {'FileId': file_id, 'header': df_headers, 'data': data})

# Vue pour afficher un tableau avec les données du fichier
def show_table(request, file_id):
    try:
        data = []
        uploaded_file = Uploaded_File.objects.get(file_id=file_id)
        file_path = uploaded_file.file.path

        # Charge le fichier CSV et nettoie les données
        df = pd.read_csv(file_path, encoding='latin1')
        df_headers = df.columns.tolist()
        df_clean = df.dropna()  # Supprime les lignes avec des valeurs nulles
        data = df_clean.to_dict(orient='records')

        return JsonResponse({'header': df_headers, 'data': data})  # Retourne les données au format JSON

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Fonction pour charger les données d'un fichier spécifique
def load_data(file_id):
    try:
        uploaded_file = Uploaded_File.objects.get(file_id=file_id)
        file_path = uploaded_file.file.path
        data = pd.read_csv(file_path)  # Charge le fichier CSV
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Convertit la colonne Date en type datetime
        data['Year'] = data['Date'].dt.year  # Extrait l'année
        data['Month'] = data['Date'].dt.month  # Extrait le mois
        data['Week'] = data['Date'].dt.isocalendar().week  # Extrait la semaine de l'année
        return data
    except Uploaded_File.DoesNotExist:
        print("Fichier introuvable.")
        return None
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
        return None

# Fonction asynchrone pour charger les données
def load_data_sync(file_id):
    return sync_to_async(load_data)(file_id)  # Appelle la fonction de manière asynchrone

# Vue pour l'analyse exploratoire des données (EDA)
async def eda_view(request, file_id):
    data = await load_data_sync(file_id)  # Charge les données de manière asynchrone

    if data is None:
        return JsonResponse({'error': 'Impossible de charger les données.'}, status=400)

    try:
        # Crée un graphique de tendance des ventes hebdomadaires
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data['Date'], y=data['Weekly_Sales'], label="Tendance des ventes")
        plt.title("Tendance des ventes au fil du temps")
        plt.xlabel("Date")
        plt.ylabel("Ventes hebdomadaires")
        plt.legend()

        # Statistiques descriptives des données
        summary_stats = data.describe().to_dict()

        # Remplace les valeurs NaN par None pour les rendre sérialisables en JSON
        summary_stats = {
            key: {sub_key: (None if isinstance(sub_value, float) and np.isnan(sub_value) else sub_value)
                  for sub_key, sub_value in value.items()}
            for key, value in summary_stats.items()
        }

        # Sauvegarde le graphique en tant qu'image base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()

        return JsonResponse({'chart': image_base64, 'summary_stats': summary_stats})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Vue pour afficher une heatmap des corrélations
async def correlation_view(request, file_id):
    data = await load_data_sync(file_id)
    feature_columns = ['Holiday_Flag', 'Temperature']  # Colonnes à analyser
    target_column = 'Weekly_Sales'  # Colonne cible

    # Génère une heatmap des corrélations
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[feature_columns + [target_column]].corr(), annot=True, cmap="coolwarm")
    plt.title("Carte de corrélation")

    # Sauvegarde le graphique en tant qu'image base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()

    return JsonResponse({'chart': image_base64})

# Vue pour entraîner un modèle et afficher les résultats
async def training_view(request, file_id):
    try:
        # Charge les données
        data = await load_data_sync(file_id)

        target_column = "Weekly_Sales"  # Colonne cible
        feature_columns = ["Holiday_Flag", "Temperature"]  # Caractéristiques

        # Prépare les données
        X = data[feature_columns + ['Year', 'Month', 'Week']]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraîne le modèle
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Fait des prédictions
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Crée un graphique des prédictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values[:50], label="Réel", marker="o")
        plt.plot(y_pred[:50], label="Prédit", marker="x")
        plt.title("Ventes réelles vs prédites")
        plt.xlabel("Échantillon")
        plt.ylabel("Ventes")
        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()

        return JsonResponse({'chart': image_base64, 'rmse': rmse, 'mae': mae})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

async def TSF_ARIMA(request, file_id):
    # Définit le chemin vers les données CSV
    file_path = "media/uploads/Walmart_Sales.csv"  # Assurez-vous que ce chemin est correct

    try:
        # Charge les données
        data = await load_data_sync(file_id)
        target_column = "Weekly_Sales"  # Colonne cible (par exemple, ventes hebdomadaires)
        date_column = "Date"

        # Filtre les données pour un magasin spécifique (modifiable selon les besoins)
        store_id = 1  # Utilise l'ID 1 par défaut si non spécifié
        store_data = data[data['Store'] == store_id] if 'Store' in data.columns else data
        store_data = store_data.sort_values(by=date_column)

        # Modèle ARIMA
        arima_model = ARIMA(store_data[target_column], order=(5, 1, 0))
        arima_result = arima_model.fit()

        # Prévisions pour les 12 prochaines semaines
        forecast = arima_result.forecast(steps=12)

        # Crée un graphique des prévisions
        plt.figure(figsize=(12, 6))
        plt.plot(store_data[date_column], store_data[target_column], label="Ventes historiques")
        plt.plot(pd.date_range(store_data[date_column].iloc[-1], periods=12, freq='W'), forecast, label="Ventes prédites",
                 color='red')
        plt.title("Prévisions des ventes")
        plt.xlabel("Date")
        plt.ylabel("Ventes")
        plt.legend()

        # Sauvegarde le graphique dans un flux BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode le graphique en base64
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        # Prépare les données de réponse pour le frontend
        response_data = {
            'forecast': forecast.tolist(),
            'image': img_str,
            'dates': pd.date_range(store_data[date_column].iloc[-1], periods=12, freq='W').strftime('%Y-%m-%d').tolist()
        }

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

