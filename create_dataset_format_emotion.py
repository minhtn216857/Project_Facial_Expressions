import os
import shutil
import pandas as pd
import cv2
import math

def Chia_train_test(is_train, data, output_dir):
    data_ap = []
    if is_train == 'train':
        for i in range(0, math.ceil(len(data) / 100)):
            start_index = i * 100
            end_index = min(start_index + 80, len(data) + 1)
            df_test = data.iloc[start_index:end_index]
            data_ap.append(df_test)

    else:
        for i in range(0, math.ceil(len(data) / 100)):
            start_index = i * 100 + 80
            end_index = min(start_index + 20, (i + 1) * 100)
            df_test = data.iloc[start_index:end_index]
            data_ap.append(df_test)
    data = pd.concat(data_ap)

    os.makedirs(os.path.join(output_dir, 'images', is_train))
    os.makedirs(os.path.join(output_dir, 'labels', is_train))

    data_images = os.path.join(output_dir, 'images', is_train) + '/' + data.iloc[:, 0].astype(str)
    data_images = data_images.tolist()
    data_image_root = os.path.join(root, 'image', 'origin') + '/' + data.iloc[:, 0].astype(str)
    data_image_root = data_image_root.tolist()
    # data_image_root_const = set(data_image_root)
    image_names = data.iloc[:, 0].astype(str)
    image_names = image_names.tolist()
    image_names = [image_name.replace('.jpg', '.txt') for image_name in image_names]

    labels = data.iloc[:, 7].astype(int)
    labels = labels.tolist()

    ymin = data.iloc[:, 2].astype(int).tolist()
    xmin = data.iloc[:, 3].astype(int).tolist()
    xmax = data.iloc[:, 4].astype(int).tolist()
    ymax = data.iloc[:, 5].astype(int).tolist()

    for i, data_image in enumerate(data_images):
        img = cv2.imread(data_image_root[i])
        # # img = img[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        # cv2.imshow('img', img)
        if not os.path.exists(data_image):
            cv2.imwrite(data_image, img)

        with open(os.path.join(output_dir, 'labels', is_train, '{}'.format(image_names[i])), 'a') as text_file:
            glob_height, glob_width, _ = img.shape
            xcent = (xmin[i] + xmax[i]) / (2 * glob_width)
            ycent = (ymin[i] + ymax[i]) / (2 * glob_height)
            width = (xmax[i] - xmin[i]) / glob_width
            height = (ymax[i] - ymin[i]) / glob_height
            text_file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(labels[i], xcent, ycent, width, height))

if __name__ == '__main__':
    root = "data_emotion"
    output_dir = 'emotion_yolo'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))
    os.makedirs(os.path.join(output_dir, 'labels'))
    image_path = os.path.join(root, 'image', 'origin')
    label_path = os.path.join(root, 'label', 'label.lst')

    data = pd.read_csv(label_path, sep=' ', header=None)
    # data = data.sample(frac=0.5)
    df1 = data[data.iloc[:, 7].isin([3, 6])]
    df1 = df1.iloc[: int(len(df1) * 0.3)]
    # print(len(df))
    df2 = data[data.iloc[:, 7].isin([0, 1, 2, 4, 5])]
    df2 = df2.iloc[: int(len(df2) * 0.9)]
    data = pd.concat([df1, df2])
    ########################################################
    Chia_train_test('train', data, output_dir)
    Chia_train_test('val', data, output_dir)

# amazed_actor_155