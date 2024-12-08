from django.db import models

# Create your models here.


class Uploaded_File(models.Model):
    file_id = models.CharField(max_length=255,blank=True)
    file_name = models.CharField(max_length=255,blank=True)
    file_format = models.CharField(max_length=255,blank=True)
    file = models.FileField(upload_to=f'uploads',blank=True,)
    file_date  = models.DateTimeField(auto_now=True,null=True,blank=True)