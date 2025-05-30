import os
import random
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split




# functions

def remove_black_borders(img):


    # remove black border
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y + h, x:x + w]
    return img

def slice(image_path, output_dir):
    slices = 3
    filename_list = []

    img = cv2.imread(image_path)
    if img is None:
        print("Image couldn't be load")
        return []



    img = remove_black_borders(img)

    height, width, _ = img.shape
    slice_width = width // slices

    for i in range(slices):
        cropped_img = img[:, i * slice_width:(i + 1) * slice_width]
        filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.jpg")
        cv2.imwrite(filename, cropped_img)
        filename_list.append(filename)
    return filename_list


def move_images_to_folders(image_list, target_dir):
    for img_path, label in image_list:
        label_dir = os.path.join(target_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(img_path, label_dir)





if __name__=="__main__":
    data_dir = "C:/Users/Msi katana/Desktop/images"
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    processed_data_dir = "dataset/processed_images"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # Process images and split data
    all_images_list = []
    for country in os.listdir(data_dir):
        print(country)
        country_path = os.path.join(data_dir, country)
        if os.path.isdir(country_path):
            processed_country_path = os.path.join(processed_data_dir, country)
            os.makedirs(processed_country_path, exist_ok=True)
            for img_file in os.listdir(country_path):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(country_path, img_file)
                    processed_images = slice(img_path, processed_country_path)
                    all_images_list.extend([(img, country) for img in processed_images])

    #split
    random.shuffle(all_images_list)
    train_images, test_images = train_test_split(all_images_list, test_size=0.2, stratify=[label for _, label in all_images_list])

    #move
    move_images_to_folders(train_images, train_dir)
    move_images_to_folders(test_images, test_dir)



