from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.models import User
from django.http import JsonResponse,Http404

from .models import Uploaded_File

import os
import random
import string
import matplotlib
matplotlib.use('Agg')
 
import seaborn as sns
from django.conf import settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings

from tempfile import NamedTemporaryFile
from asgiref.sync import sync_to_async
from io import BytesIO
import base64

import json

def Home(request):
    files_list= Uploaded_File.objects.all()
    return render(request,'html/main.html',{'files_list':files_list,})


def generate_unique_file_id():
    while True:
        file_id = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=10))
        if not Uploaded_File.objects.filter(file_id=file_id).exists():
            return file_id

def Upload_File(request):
    if request.method == "POST":
        upFile = request.FILES.get('file')

        if upFile:
            file_name = upFile.name
            file_format = file_name.split('.')[-1]

            # Generate a unique fileId
            fileId = generate_unique_file_id()

            # Create the Uploaded_File object
            Uploaded_File.objects.create(
                file_id=fileId,
                file_name=file_name,
                file_format=file_format,
                file=upFile
            )

            return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error'})



def remove_file(request, file_id):
    try:
        # Try to fetch the file object with the given file_id
        file_to_remove = Uploaded_File.objects.get(file_id=file_id)

        # Delete the file from the media directory if it exists
        if file_to_remove.file:
            file_to_remove.file.delete(save=False)  # This will delete the file from the file system
        
        # Delete the file record from the database
        file_to_remove.delete()

        # Redirect to the home page
        return redirect('home')  # Redirect to the home page

    except Uploaded_File.DoesNotExist:
        # If the file does not exist, raise an error
        raise Http404("File not found")




def MAIN(request,file_id):
    data=[]
    uploaded_file = Uploaded_File.objects.get(file_id=file_id)
    file_path = uploaded_file.file.path
    
    df = pd.read_csv(file_path,encoding='latin1')
    df_headers = df.columns.tolist()
    df_clean = df.loc[~df.isnull().any(axis=1)]
    data=df_clean.to_dict(orient='records')

        
    return render(request,'html/dataPage.html',{'FileId':file_id,'header':df_headers,
                            'data':data})



def show_table(request, file_id):
    try:
        data = []
        uploaded_file = Uploaded_File.objects.get(file_id=file_id)
        file_path = uploaded_file.file.path
        
        # Load CSV and clean data
        df = pd.read_csv(file_path, encoding='latin1')
        df_headers = df.columns.tolist()
        df_clean = df.dropna()  # Drop rows with NaN values
        data = df_clean.to_dict(orient='records')

        return JsonResponse({'header': df_headers, 'data': data})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

# Define the synchronous load_data function
def load_data(file_id,target_column):
    try:
        uploaded_file = Uploaded_File.objects.get(file_id=file_id)
        file_path = uploaded_file.file.path
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Week'] = data['Date'].dt.isocalendar().week
        # Adding Lag Feature for Sales (previous week's sales)
        data['Lagged_Sales'] = data[target_column].shift(1)

        # Rolling Statistics (e.g., rolling mean over 4 weeks)
        data['Rolling_Mean_S'] = data[target_column].rolling(window=4).mean()

        # Drop rows with missing values after feature engineering
        data = data.dropna()
        
        return data
    except Uploaded_File.DoesNotExist:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Use sync_to_async to call the synchronous function
async def load_data_sync(file_id,target_column):
    return await sync_to_async(load_data)(file_id,target_column)


async def eda_view(request, file_id):
      # Load your dataframe asynchronously
    
    date_column = request.GET.get('date_column')
    target_column = request.GET.get('target_column')
    data = await load_data_sync(file_id,target_column)
    if data is None:
        return JsonResponse({'error': 'Data could not be loaded.'}, status=400)

    try:
        # Create a line plot for Weekly Sales
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data[date_column], y=data[target_column], label=f"{target_column} Trend")
        plt.title(f"{target_column} Over Time")
        plt.xlabel("Date")
        plt.ylabel(target_column)
        plt.legend()

        # Summary statistics of the data
        summary_stats = data.describe().to_dict()

        # Replace NaN values with None to make the data JSON serializable
        summary_stats = {
            key: {sub_key: (None if isinstance(sub_value, float) and np.isnan(sub_value) else sub_value)
                  for sub_key, sub_value in value.items()}
            for key, value in summary_stats.items()
        }

        # Save the plot to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()

        return JsonResponse({'chart': image_base64,
                             'summary_stats': summary_stats})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Correlation View
async def correlation_view(request,file_id):
    
    feature_columns = json.loads(request.GET.get('feature_columns', '[]'))

    date_column = request.GET.get('date_column')
    target_column = request.GET.get('target_column')
    data = await load_data_sync(file_id,target_column)
    
    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[feature_columns + [target_column]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    # Save the plot to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()

    return JsonResponse({'chart': image_base64})

# Training Model
async def training_view(request, file_id):
    try:
        # Load data

        feature_columns = json.loads(request.GET.get('feature_columns', '[]'))
        date_column = request.GET.get('date_column')
        target_column = request.GET.get('target_column')
        model_name = request.GET.get('model_name')  # Default to Random Forest

        data = await load_data_sync(file_id,target_column)
        # Prepare data
        X = data[feature_columns + ['Year', 'Month', 'Week', 'Lagged_Sales']]
        y = data[target_column]

        # Train-test split using TimeSeriesSplit (to preserve temporal order)
        tscv = TimeSeriesSplit(n_splits=5)

        # Model selection
        if model_name == 'random_forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            }
        elif model_name == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            param_grid = {}  # No hyperparameters to tune for Linear Regression
        elif model_name == 'xgboost':
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            return JsonResponse({'error': 'Invalid model name'}, status=400)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        # Predictions and metrics
        y_pred = grid_search.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y[:50].values, label="Actual", marker="o")  # Actual Sales
        plt.plot(y_pred[:50], label="Predicted", marker="x")  # Predicted Sales
        plt.title("Actual vs Predicted Data")
        plt.xlabel("Sample")
        plt.ylabel(target_column)
        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()

        return JsonResponse({'chart': image_base64, 'rmse': rmse, 'mae': mae})
    except Exception as e:
        print("\n\n\nException ", e, "\n\n\n")
        return JsonResponse({'error': str(e)}, status=500)
    

async def TSF_ARIMA(request,file_id):

    try:
        # Load dataset
        date_column = request.GET.get('date_column')
        target_column = request.GET.get('target_column')
        
        data = await load_data_sync(file_id,target_column)
        
        # Process data for specific store (you can modify this based on your requirements)
        store_id = int(request.GET.get('store_id'))  # Default to store 1 if not specified
        store_data = data[data['Store'] == store_id] if 'Store' in data.columns else data
        store_data = store_data.sort_values(by=date_column)

        # ARIMA Model
        arima_model = ARIMA(store_data[target_column], order=(5, 1, 0))
        arima_result = arima_model.fit()

        # Forecast Next 12 Weeks
        forecast = arima_result.forecast(steps=12)

        # Plot Forecast
        plt.figure(figsize=(12, 6))
        plt.plot(store_data[date_column], store_data[target_column], label="Historical "+target_column)
        plt.plot(pd.date_range(store_data[date_column].iloc[-1], periods=12, freq='W'), forecast, label="Forecasted "+target_column,
                color='red')
        plt.title(target_column+" Forecast")
        plt.xlabel("Date")
        plt.ylabel(target_column)
        plt.legend()

        # Save the plot as a PNG in a BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the image as base64
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        # Prepare the result to send back to the frontend
        response_data = {
            'forecast': forecast.tolist(),
            'image': img_str,
            'dates': pd.date_range(store_data[date_column].iloc[-1], periods=12, freq='W').strftime('%Y-%m-%d').tolist()
        }

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

