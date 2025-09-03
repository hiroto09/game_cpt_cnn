import cv2
import time
import datetime
import numpy as np
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv() 

# ---- CNN モデルの準備 ----
model = load_model("./saved_model/game_classifier.h5")
class_names = ["人生ゲーム", "スマブラ"]  # dataset のフォルダ名に合わせる

# ---- キャプチャーボードを開く ----
capture = cv2.VideoCapture(0)  # 環境に応じて 0,1,2 を変更
if not capture.isOpened():
    print("キャプチャーボードが開けませんでした")
    exit()

interval = 60  # 1分ごと
last_pred_time = time.time()

# ---- POST 先の API ----
api_url = os.getenv("API_URL")  # 環境変数から取得
if not api_url:
    print("API_URL が設定されていません")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("映像を取得できませんでした")
        break

    now = time.time()
    if now - last_pred_time >= interval:
        # ---- 前処理（128x128, 正規化）----
        img_resized = cv2.resize(frame, (128, 128))
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)  # (1,128,128,3)

        # ---- CNN 推論 ----
        pred = model.predict(img_input)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        result = {
            "predicted_class": class_names[class_id],
            "confidence": confidence,
            "timestamp": datetime.datetime.now().isoformat()
        }
        print("推論結果:", result)

        # ---- API に送信 ----
        try:
            response = requests.post(api_url, json=result, timeout=10)
            if response.status_code == 200:
                print("API 送信成功:", response.json())
            else:
                print("API エラー:", response.status_code, response.text)
        except Exception as e:
            print("API 送信エラー:", e)

        last_pred_time = now

    # ---- 映像を表示したい場合 ----
    cv2.imshow("Capture", frame)

    # 'q' キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
