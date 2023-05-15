from django.contrib import admin
from .models import PrintLog

@admin.register(PrintLog)
class LogAdmin(admin.ModelAdmin):
    list_display = ('id','wav_file', 'image', 'result', 'create_dt', 'update_dt')