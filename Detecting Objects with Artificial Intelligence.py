import cv2  # OpenCV 라이브러리 import (영상 처리 및 딥러닝 기능 사용)

# 모델이 인식할 수 있는 객체 클래스 목록 (ID : 이름)
classNames = {0: 'background',
1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes): # class ID를 이름으로 변환하는 함수
    for key, value in classes.items():  # 클래스 목록을 순회
        if class_id == key:  # 입력된 ID와 일치하는 경우
            return value  # 해당 객체 이름 반환

# 사전 학습된 TensorFlow 모델 로드 (.pb: 가중치, .pbtxt: 구조)
model = cv2.dnn.readNetFromTensorflow(
    'models/frozen_inference_graph.pb',
    'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
)


image = cv2.imread("image.jpeg")    # 입력 이미지 불러오기


image_height, image_width, _ = image.shape  # 이미지 크기 저장 (박스 좌표 계산에 사용)


model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))  # 이미지를 모델 입력 형태로 변환하여 설정 (Blob 변환 포함)


output = model.forward()   # 모델을 통해 객체 검출 수행 (순전파)


for detection in output[0, 0, :, :]:  # 검출된 객체들을 하나씩 확인

    confidence = detection[2]  # 객체 인식 신뢰도


    if confidence > .5:       # 신뢰도가 0.5 이상인 경우만 사용

        class_id = detection[1]  # 객체 클래스 ID

       
        class_name = id_class_name(class_id, classNames)   # ID를 실제 객체 이름으로 변환

       
        print(str(str(class_id) + " " + str(detection[2]) + " " + class_name))   # 결과 출력 (ID, 신뢰도, 이름)

        # 바운딩 박스 좌표 계산 (이미지 크기 기준 변환)
        box_x = detection[3] * image_width 
        box_y = detection[4] * image_height 
        box_width = detection[5] * image_width 
        box_height = detection[6] * image_height 

        # 객체 위치를 사각형으로 표시
        cv2.rectangle(
            image,
            (int(box_x), int(box_y)),
            (int(box_width), int(box_height)),
            (23, 230, 210),
            thickness=1
        )

        # 객체 이름을 이미지 위에 표시
        cv2.putText(
            image,
            class_name,
            (int(box_x), int(box_y + .05 * image_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            (.005 * image_width),
            (0, 0, 255)
        )

# 결과 이미지 화면 출력
cv2.imshow('image', image)

# 키 입력 대기 (0이면 무한 대기)
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()