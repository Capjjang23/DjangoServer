from PIL import Image
from django.http import HttpResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import io

import requests
import traceback
import json


def get_spectrogram(request):
    if request.method == 'GET':
        # 클라이언트에서 전송한 파일을 가져옵니다.

        # response = None  # 초기값을 지정해줍니다.
        audio_path = 'djangoServer/audio/m.m4a'
        # url = 'http://localhost:8000/process_audio/'
        url = 'http://223.194.153.133:8000/process_audio/'
        try:
            with open(audio_path, 'rb') as f:
                # audio_file = {'m4a' : f}
                byte_array = bytearray(f.read())  # 파일을 바이트 배열로 읽음
                # print(byte_array)
                if not f.closed:
                    print(f"{audio_path} is opened.")
                # spectrogram = requests.post(url, files={'m4a': f})
                response = requests.post(url, data=byte_array)
                print(response.status_code)
                print(response.text)
                # print(spectrogram)

                if response.status_code == 200:
                    response_data = json.loads(response.content.decode('utf-8')) if response else {}
                    print(response_data)

                    predicted_alphabet = response_data['predicted_alphabet']
                    return JsonResponse({'predicted_alphabet': predicted_alphabet})
                else:
                    print(f"An error occurred: {response.status_code}")
                    return JsonResponse({'error': f"An error occurred: {response.status_code}"})

        except FileNotFoundError:
            # 파일이 존재하지 않을 때의 예외 처리
            print(f"{audio_path} does not exist.")
            return JsonResponse({'error': f"An error occurred: FileNotFoundError"})
        except Exception as e:
            # 그 외의 예외 처리
            print(f"An error occurred while opening {audio_path}: {e}")
            return JsonResponse({'error': f"An error occurred: FileNotFoundError"})


# POST 응답 처리
from pydub import AudioSegment
import torch
import torchvision.transforms as transforms
import numpy as np
import librosa, librosa.display
import asyncio
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from django.conf import settings

FIG_SIZE = (15, 10)
DATA_NUM = 30

alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z']


# m4a -> wav -> spectrogram / -> model -> result
@csrf_exempt
def process_audio(request):
    global peekIndex, image_url

    print("process_audio")
    try:
        if request.method == 'POST':
            print("POST")

            # -------여기가 받는 곳 -------
            # byte_array = request.body  # 안드로이드 앱에서 보낸 데이터를 가져옵니다.
            # print(byte_array)
            # #데이터 처리 로직 작성
            # response_data = {'key': 'mimifool'}  # 안드로이드 앱에게 보낼 응답 데이터를 딕셔너리 형태로 작성합니다.
            # return HttpResponse(json.dumps(response_data), content_type="application/json")

            # POST 요청에서 이미지 파일을 가져옵니다.
            # m4a_file = request.FILES['m4a']
            # print("m4a_file : ", m4a_file)

            # POST 요청에서 biteArray 데이터를 가져옵니다.
            byte_array = request.body  # 안드로이드 앱에서 보낸 데이터를 가져옵니다.
            # print("테스트: ",byte_array)
            with open('my_audio_file.wav', 'wb') as f:
                f.write(byte_array)

            # 바이트 배열을 파일처럼 BytesIO 객체를 생성한다.
            # byte_io = io.BytesIO(byte_array)
            # librosa.load 함수를 사용하여 음향 데이터를 읽어온다.
            # sig, sr = librosa.load(io.BytesIO(byte_array), sr=22050)
            # # 3초의 묵음을 추가한다.
            # silence_duration = 3  # 3초
            # silence_samples = np.zeros(int(silence_duration * sr))
            # y_with_silence = np.concatenate([sig, silence_samples]) # numpy 형태

            # y_with_silence를 다시 바이트 배열로 변환한다.
            # byte_array_with_silence = bytearray(y_with_silence)
            #
            # # 바이트 배열을 신호로 변환한다.
            # sig, sr = librosa.load(io.BytesIO(byte_array_with_silence), sr=22050)

            # 소리 + 묵음
            # load the audio files
            audio1 = AudioSegment.from_file("my_audio_file.wav", format="wav")
            #audio2 = AudioSegment.from_file("djangoServer/slienceSound.m4a", format="m4a")
            silence = AudioSegment.silent(duration=3000) #3초 묵음


            # concatenate the audio files
            #combined_audio = audio1 + audio2
            combined_audio = audio1+silence

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
            #
            START_LEN = 1102
            END_LEN = 20948
            if peekIndex > 1102:
                print(peekIndex)
                startPoint = peekIndex - START_LEN
                endPoint = peekIndex + 22050
            else:
                print(peekIndex)
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
            # stft = librosa.stft(sig[startPoint:endPoint], n_fft=n_fft, hop_length=hop_length)
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
            # spectrogram 이미지 저장
            image_path = 'static/images/' + 'test.jpg'
            plt.savefig(image_path)

            plt.close()

            # 모델 입히기
            # load the saved ResNet model
            model = torch.load('djangoServer/model/resnet32.pth')
            # switch model to evaluation mode
            model.eval()
            # define the image transforms
            image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # 이미지 열기
            image = Image.open(image_path)

            # apply the transforms to the test image
            test_image_tensor = image_transforms(image)
            # add batch dimension to the image tensor
            test_image_tensor = test_image_tensor.unsqueeze(0)

            # get the model's prediction
            with torch.no_grad():
                prediction = model(test_image_tensor)

            # get the predicted class index
            predicted_class_index = torch.argmax(prediction).item()

            # 예측값 알파벳 출력
            print(alpha[predicted_class_index])

            # 이미지를 바이트 형태로 변환하여 메모리에 저장
            # image_bytes = BytesIO()
            # image.save(image_bytes, format='JPEG')
            # image_bytes = image_bytes.getvalue()

            # 이미지를 HttpResponse 객체에 첨부 파일로 반환
            # response = HttpResponse(image_bytes, content_type='image/jpeg')
            # response['Content-Disposition'] = 'inline; filename="spectrogram.jpeg"'
            # return response
            response = {'predicted_alphabet': alpha[predicted_class_index]}
            print(response)
            return JsonResponse(response)
    except Exception as e:
        print(traceback.format_exc())  # 예외 발생시 traceback 메시지 출력
        return HttpResponseServerError()  # 500 Internal Server Error 응답 반환
