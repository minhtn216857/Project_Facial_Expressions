import cv2
import torch
import argparse
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
from FCNN_resnet_emotion.src.dataset import categories


def get_args():
    parser = argparse.ArgumentParser("Detector_classifier pipeline")
    parser.add_argument("--detector_checkpoint", "-d", type=str,
                        default="/home/minhtran/Documents/MDEEP_LEARNING/Project_Emotion/best.pt")
    parser.add_argument("--classifier_checkpoint", type=str,
                        default="/home/minhtran/Documents/MDEEP_LEARNING/Project_Emotion/class_save_model/best.pt")
    parser.add_argument("--video", "-v", type=str, default="")
    parser.add_argument("--image_path", "-i", type=str,
                        default="/home/minhtran/Documents/MDEEP_LEARNING/Project_Emotion/demo/livver6.jpg")
    args = parser.parse_args()
    return args

# class Emotion_resnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = models.resnet50()
#         del self.backbone.fc
#         self.backbone.fc = nn.Linear(2048, 6)
#
#     def forward(self, x):
#         # See note [TorchScript super()]
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)
#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)
#         x = self.backbone.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.backbone.fc(x)
#
#         return x


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load obj detect
    detector = torch.hub.load('/home/minhtran/Documents/MDEEP_LEARNING/yolov5-master',
                              'custom', args.detector_checkpoint,
                              source='local')  # local repo
    # Load classifier
    classifier = resnet18()
    del classifier.fc
    classifier.fc = nn.Linear(in_features=512, out_features=7)
    checkpoint = torch.load(args.classifier_checkpoint)

    classifier.load_state_dict(checkpoint["model_state_dict"])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Read image
    ori_img = cv2.imread(args.image_path)

    detector.to(device).eval()
    classifier.to(device).eval()

    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    det_pred = detector(img, size=640)

    face_images = []
    for coord in det_pred.xyxy[0]:
        xmin, ymin, xmax, ymax, _, _ = coord
        face_image = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
        face_image = transform(face_image)
        face_images.append(face_image)

    face_images = torch.stack(face_images).to(device)
    # Classifier's inference
    with torch.no_grad():
        cls_pred = classifier(face_images)
        number_class = torch.argmax(cls_pred, dim=1)
    # print(cls_pred)
    # print(number_class)
    for (xmin, ymin, xmax, ymax, _, _), n in zip(det_pred.xyxy[0], number_class):
        n = n.item()
        cv2.rectangle(ori_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
        cv2.putText(ori_img, str(categories[n]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('test', ori_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = get_args()
    inference(args)

