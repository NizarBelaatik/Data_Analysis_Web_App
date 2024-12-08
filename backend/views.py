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


    
def MAIN(request):
    return render(request,'html/main.html',{'msg':'test this',})


def Upload_File(request):
    if request.method == "POST":
        upFile = request.FILES.get('file')

        if upFile:
            file_name=upFile.name
            file_format= file_name.split('.')[-1]
            fileId= ''
            Uploaded_File.objects.create(
                                        file_id=fileId,
                                        file_name=file_name,
                                         file_format=file_format,
                                        file=upFile)
        
            return JsonResponse({'status':'success',})
    return JsonResponse({'status':'error',})


