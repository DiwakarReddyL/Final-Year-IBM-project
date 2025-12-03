from django.db import models
from django.contrib.auth.models import User

class UploadedCSV(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='csvs/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def filename(self):
        return self.file.name.split('/')[-1]

    def __str__(self):
        return f"{self.user.email} - {self.filename()}"
