
import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, cv2
import numpy as np

headers = {
    # リクエストヘッダー
    'Content-Type': 'application/octet-stream',
    # Azure Face APIのキーを指定
    'Ocp-Apim-Subscription-Key': '{ Key }',
}

params = urllib.parse.urlencode({
    # リクエストパラメータ
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
})

# Azure Face APIのエンドポイント設定
face_api_url = '{Azure Face API Endpoint URL}/detect'

# 読み込む画像(複数)
image_urls = ['./images/hoge.jpg','./images/fuga.jpg']
colors = {'happiness': (0,255,0), 'disgust': (255, 0,0), 'anger': (0,0,255), 'surprise': (255,0,255), 'neutral': (255,255,255), 'sadness': (255,255,0), 'contempt': (0,255,255), 'fear': (0,0,0)}

for image_url in image_urls:
    image_data = open(image_url , 'rb')
    img = cv2.imread(image_url , 1)

    try:
        response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
        datas = response.json()

        for i, data in enumerate(datas):
            # 顔の位置を四角で囲む
            faceRectangle = data['faceRectangle']
            x_max = faceRectangle['left'] + faceRectangle['width']
            x_min = faceRectangle['left']
            y_max = faceRectangle['top'] + faceRectangle['height']
            y_min = faceRectangle['top']
            
            # 一番近い表情とその数値を表示(各四角の上に)
            emotions = data['faceAttributes']['emotion']
            top_emotion = {'emotion':'empty', 'value':0}
            emotion_value = 0
            for e in emotions:
                emotion_value = emotions[e]
                if emotion_value >= top_emotion['value']:
                    top_emotion['emotion'] = e
                    top_emotion['value'] = emotion_value
            color = colors[top_emotion['emotion']]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 7)
            text = top_emotion['emotion'] + ': ' + str(top_emotion['value']*100) + '%'
            
            cv2.putText(img, text, (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
            
            print(data)
        cv2.imwrite('test_' + data['faceId'] + '.jpg', img)
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
