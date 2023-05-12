from django.db import models

class PrintLog(models.Model):
    wav_file = models.FileField(upload_to='audio/')
    result = models.CharField(max_length=100)

    def __str__(self):
        return self.wav_file.name