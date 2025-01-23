from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.models import User
from django.http import JsonResponse,Http404

from .models import Uploaded_File

import pandas as pd
import os
import random
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
import seaborn as sns
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


from tempfile import NamedTemporaryFile
from asgiref.sync import sync_to_async
from io import BytesIO
import base64


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

        
    return render(request,'html/chart.html',{'FileId':file_id,'header':df_headers,
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
def load_data(file_id):
    try:
        uploaded_file = Uploaded_File.objects.get(file_id=file_id)
        file_path = uploaded_file.file.path
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Week'] = data['Date'].dt.isocalendar().week
        return data
    except Uploaded_File.DoesNotExist:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Use sync_to_async to call the synchronous function
async def load_data_sync(file_id):
    return await sync_to_async(load_data)(file_id)


async def eda_view(request, file_id):
    data = await load_data_sync(file_id)  # Load your dataframe asynchronously

    if data is None:
        return JsonResponse({'error': 'Data could not be loaded.'}, status=400)

    try:
        # Create a line plot for Weekly Sales
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data['Date'], y=data['Weekly_Sales'], label="Sales Trend")
        plt.title("Sales Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Weekly Sales")
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
    data = await load_data_sync(file_id)
    feature_columns = ['Holiday_Flag', 'Temperature']
    target_column = 'Weekly_Sales'

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
async def training_view(request,file_id):
    try:
        # Load data
        data = await load_data_sync(file_id)

        target_column = "Weekly_Sales"
        feature_columns = ["Holiday_Flag", "Temperature"]

        # Prepare data
        X = data[feature_columns + ['Year', 'Month', 'Week']]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values[:50], label="Actual", marker="o")
        plt.plot(y_pred[:50], label="Predicted", marker="x")
        plt.title("Actual vs Predicted Sales")
        plt.xlabel("Sample")
        plt.ylabel("Sales")
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

async def TSF_ARIMA(request,file_id):
    # Define file path to your CSV data
    file_path = "media/uploads/Walmart_Sales.csv"  # Make sure this path is correct

    # Check if the file exists
    #if not os.path.exists(file_path):
     #   return JsonResponse({"error": "Dataset file not found."}, status=404)

    try:
        # Load dataset
        data = await load_data_sync(file_id)
        target_column ="Weekly_Sales"# input("Enter the target column name (e.g., Weekly_Sales): ")
        date_column = "Date"
        
        # Process data for specific store (you can modify this based on your requirements)
        store_id = 1# int(request.GET.get('store_id', 1))  # Default to store 1 if not specified
        store_data = data[data['Store'] == store_id] if 'Store' in data.columns else data
        store_data = store_data.sort_values(by=date_column)

        # ARIMA Model
        arima_model = ARIMA(store_data[target_column], order=(5, 1, 0))
        arima_result = arima_model.fit()

        # Forecast Next 12 Weeks
        forecast = arima_result.forecast(steps=12)

        # Plot Forecast
        plt.figure(figsize=(12, 6))
        plt.plot(store_data[date_column], store_data[target_column], label="Historical Sales")
        plt.plot(pd.date_range(store_data[date_column].iloc[-1], periods=12, freq='W'), forecast, label="Forecasted Sales",
                color='red')
        plt.title("Sales Forecast")
        plt.xlabel("Date")
        plt.ylabel("Sales")
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

