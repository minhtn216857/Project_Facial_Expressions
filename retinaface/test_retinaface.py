import cv2
from retinaface.pre_trained_models import get_model
from pprint import pprint


model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()
img = cv2.imread("/home/minhtran/Documents/MDEEP_LEARNING/Project_Emotion/demo/liver3.jpg")
# print(img.shape)
annotation = model.predict_jsons(img)
# pprint(annotation)
for face in annotation:
    # print((face['bbox']))
    xmin, ymin, xmax, ymax = face['bbox']
    # print(xmin)
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
    cv2.putText(img, 'face', (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("test", img)
cv2.waitKey(0)