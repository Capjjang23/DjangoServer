from django.db import models

class PrintLog(models.Model):
    wav_file = models.FileField(upload_to='audio/')
    image = models.ImageField('IMAGE',upload_to='specImage/%Y/%m/',blank=True,null=True)
    result = models.CharField(max_length=50)
    create_dt = models.DateTimeField('CREATE DT', auto_now_add=True)
    update_dt = models.DateTimeField('UPDATE DT', auto_now=True)

    def __str__(self):
        return self.wav_file.name