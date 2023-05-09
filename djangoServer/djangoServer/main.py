from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework.response import Response

#
# class FileUploadView(APIView):
#     parser_class = (FileUploadParser,)
#
#     def post(self, request, format=None):
#         file_obj = request.data['file']
#         # file_obj를 처리하는 코드 작성
#         return Response(status=204)

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()


# POST 요청을 시뮬레이션하기 위한 코드
class Request:
    def __init__(self, filename):
        self.method = 'POST'
        self.FILES = {'m4a': filename}


import requests

url = 'http://172.30.1.17:8000/post_image/' # Django 서버 URL
filename = '../m4a/z.m4a'
files = {'m4a': open(filename, 'rb')} # 파일 열기

response = requests.post(url, files=files) # POST 요청 보내기

print(response.json()) # Django 서버에서 반환한 JSON 출력하기


