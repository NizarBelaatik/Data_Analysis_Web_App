"""
URL configuration for M1_S_F project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static

from backend.views import MAIN , Upload_File,show_table_type,f1
from backend.views import MAIN2,eda_view,correlation_view,training_view,TSF_ARIMA
urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('', MAIN),
    path('upload_file/',Upload_File,name="upload_file"),
    path('show_table_type/',show_table_type,name="show_table_type"),
    
    path('f1/<str:file_name>',f1,name="f1"),


    path('MAIN2/', MAIN2, name='MAIN2'),
    path('eda_view/', eda_view, name='eda_view'),
    path('correlation_view/', correlation_view, name='correlation_view'),
    path('training_view/', training_view, name='training_view'),
    #path('tsf_arima_view/', tsf_arima_view, name='tsf_arima_view'),
    
    
    #path('EDA_view2/', EDA_view2, name='EDA_view2'),
    path('TSF_ARIMA/', TSF_ARIMA, name='TSF_ARIMA'),



    ]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT)