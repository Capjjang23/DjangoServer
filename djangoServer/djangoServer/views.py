import json
import os
from io import BytesIO

from PIL.Image import Image
from django.http import HttpResponse, HttpResponseNotAllowed
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import requests

#GET 요청을 처리
# def get_audio(request):
#     if request.method == 'GET':
#         # GET 요청에서 파일 이름을 가져옵니다.
#         file_name = request.GET.get('file_name')
#
#         # 파일을 열어서 바이너리 응답으로 반환합니다.
#         with open(file_name, 'rb') as f:
#             response = HttpResponse(f.read(), content_type='m4a/*')
#             response['Content-Disposition'] = 'attachment; filename=' + file_name
#             return response
#
# url = 'http://localhost:8000/get_audio'
# params = {'file_name': 'z.m4a'}
# response = requests.get(url, params=params)



def get_spectrogram(request):
    if request.method == 'GET':
        # 클라이언트에서 전송한 파일을 가져옵니다.
        audio_file = request.FILES.get('audio_file')

        # 스펙트로그램 이미지를 생성합니다.
        spectrogram = process_audio(audio_file)

        # 스펙트로그램 이미지를 응답으로 반환합니다.
        response = HttpResponse(spectrogram, content_type='image/png')
        return response



# POST 응답 처리
from pydub import AudioSegment
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.conf import settings


FIG_SIZE = (15, 10)
DATA_NUM = 30

# m4a -> wav -> spectrogram / -> model -> result
@csrf_exempt
def process_audio(request):
    if request.method == 'POST':
        # POST 요청에서 이미지 파일을 가져옵니다.
        m4a_file = request.FILES['m4a']

        # 소리 + 묵음
        # load the audio files
        audio1 = AudioSegment.from_file(m4a_file, format="m4a")
        audio2 = AudioSegment.from_file("silenceSound.m4a", format="m4a")

        # concatenate the audio files
        combined_audio = audio1 + audio2

        # export the concatenated audio as a new file
        file_handle = combined_audio.export("combined.wav", format="wav")


        # paths.append(file_path)
        sig, sr = librosa.load(file_handle, sr=22050)

        # 에너지 평균 구하기
        sum = 0
        for i in range(0, sig.shape[0]):
            sum += sig[i] ** 2
        mean = sum / sig.shape[0]

        # 피크인덱스 찾기
        for i in range(0, sig.shape[0]):
            if (sig[i] ** 2 >= mean):
                peekIndex = i
                break

        START_LEN = 1102
        END_LEN = 20948
        if peekIndex > 1102:
            startPoint = peekIndex - START_LEN
            endPoint = peekIndex + 22050
        else:
            startPoint = peekIndex
            endPoint = peekIndex + END_LEN

            # 단순 푸리에 변환 -> Specturm
            fft = np.fft.fft(sig[startPoint:endPoint])

            # 복소공간 값 절댓갑 취해서, magnitude 구하기
            magnitude = np.abs(fft)

            # Frequency 값 만들기
            f = np.linspace(0, sr, len(magnitude))

            # 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날려고 앞쪽 절반만 사용한다.
            left_spectrum = magnitude[:int(len(magnitude) / 2)]
            left_f = f[:int(len(magnitude) / 2)]

            # STFT -> Spectrogram
            hop_length = 512  # 전체 frame 수
            n_fft = 2048  # frame 하나당 sample 수

            # calculate duration hop length and window in seconds
            hop_length_duration = float(hop_length) / sr
            n_fft_duration = float(n_fft) / sr

            # STFT
            stft = librosa.stft(sig[startPoint:endPoint], n_fft=n_fft, hop_length=hop_length)

            # 복소공간 값 절댓값 취하기
            magnitude = np.abs(stft)

            # magnitude > Decibels
            log_spectrogram = librosa.amplitude_to_db(magnitude)

            FIG_SIZE = (10, 10)

            # display spectrogram
            plt.figure(figsize=FIG_SIZE)
            librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, cmap='magma')

            # matplotlib 라이브러리를 사용하여 생성된 spectrogram 이미지를 jpg 형식으로 저장
            name_end_pos = file_handle.find('.')
            image_path = '/images/' + file_handle[:name_end_pos] + '.jpg'
            image_url = settings.STATIC_URL + image_path

            # save spectrogram image
            plt.savefig('static/images/' + file_handle[:name_end_pos] + '.jpg')
            plt.close()


        # 이미지 열기
        image = Image.open(image_url)

        # 이미지를 바이트 형태로 변환하여 메모리에 저장
        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()

        # 이미지를 HttpResponse 객체에 첨부 파일로 반환
        response = HttpResponse(image_bytes, content_type='image/jpeg')
        response['Content-Disposition'] = 'attachment; filename="spectrogram.jpg"'
        return response


url = 'http://localhost:8000/get_spectrogram'
files = {'audio_file': open('djangoServer/djangoServer/z.m4a', 'rb')}
response = requests.get(url, files=files)

# 응답으로 받은 스펙트로그램 이미지를 저장합니다.
with open('spectrogram.png', 'wb') as f:
    f.write(response.content)





