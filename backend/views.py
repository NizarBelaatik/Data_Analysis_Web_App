from django.shortcuts import render
from django.template.loader import render_to_string
from django.shortcuts import redirect , get_object_or_404

from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login , logout,update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt,csrf_protect
from django.http import JsonResponse
from django.core import serializers


from .models import Uploaded_File

import pandas as pd
#import numpy as np

import os
import random
import string

    
def MAIN(request):
    files_list= Uploaded_File.objects.all()
    return render(request,'html/main.html',{'files_list':files_list,})


def Upload_File(request):
    if request.method == "POST":
        upFile = request.FILES.get('file')

        if upFile:
            file_name=upFile.name
            file_format= file_name.split('.')[-1]
            
            fileId= ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            Uploaded_File.objects.create(
                                        file_id=fileId,
                                        file_name=file_name,
                                         file_format=file_format,
                                        file=upFile)
        
            return JsonResponse({'status':'success',})
    return JsonResponse({'status':'error',})



def preProc(request):
    if request.method == 'POST':
        df = pd.read_csv('media\uploads\Employee_Sample_Data.csv',encoding='latin1')
        df_headers = df.columns.to_dict()
        df_clean = df.loc[df.isnull().any(axis=1)]
        
        print(df_clean)
        html=render_to_string({'header':df_headers,
                            'data':df_clean})
        return JsonResponse({'status':'success',
                            
                            'html':html})
        
    return JsonResponse({'status':'error',})