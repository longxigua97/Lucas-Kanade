import numpy as np
import cv2
import matplotlib.pyplot as plt

file_path = "video/D0002060316_00000_V_000.mp4"

# 動画ファイルのロード
video = cv2.VideoCapture(file_path)

# 150フレームから210フレームまで5フレームごとに切り出す
start_frame = 150
end_frame = 210
interval_frames = 5
i = start_frame + interval_frames

# 最初のフレームに移動して取得
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ret, prev_frame = video.read()

# グレースケールにしてコーナ特徴点を抽出
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

feature_params = {
    "maxCorners": 200,  # 特徴点の上限数
    "qualityLevel": 0.2,  # 閾値　（高いほど特徴点数は減る)
    "minDistance": 12,  # 特徴点間の距離 (近すぎる点は除外)
    "blockSize": 12  # 
}
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)


# 特徴点をプロットして可視化
for p in p0:
    x,y = p.ravel()
    #print(x,y)
    cv2.circle(prev_frame, (int(x), int(y)), 5, (0, 255, 255) , -1)

#cv2.imshow('camera', prev_frame)



# OpticalFlowのパラメータ
lk_params = {
    "winSize": (15, 15),  # 特徴点の計算に使う周辺領域サイズ
    "maxLevel": 2,  # ピラミッド数 (デフォルト0で、2の場合は1/4の画像まで使われる)
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 探索アルゴリズムの終了条件
}

# 可視化用
color = np.random.randint(0, 255, (200, 3))
mask = np.zeros_like(prev_frame)

for i in range(start_frame + interval_frames, end_frame + 1, interval_frames):
    # 次のフレームを取得してグレースケールにする
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # OpticalFlowの計算
    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
    #print(p1, status, err)
    #break
    # フレーム前後でトラックが成功した特徴点のみを

    identical_p1 = p1[status==1]
    identical_p0 = p0[status==1]
    
    print(identical_p1,p1)
    # 可視化用
    for i, (p1, p0) in enumerate(zip(identical_p1, identical_p0)):
        p1_x, p1_y = p1.ravel()
        p0_x, p0_y = p0.ravel()
        frame = cv2.line(frame, (int(p1_x), int(p1_y)), (int(p0_x), int(p0_y)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(p1_x), int(p1_y)), 5, color[i].tolist(), -1)
    
    # 可視化用の線・円を重ねて表示
    image = cv2.add(frame, mask)
    cv2.imshow('camera', frame)
    cv2.waitKey()
    # トラックが成功した特徴点のみを引き継ぐ
    prev_gray = frame_gray.copy()
    p0 = identical_p1.reshape(-1, 1, 2)
    print(p0)

#cv2.waitKey(0)
cv2.destroyAllWindows()