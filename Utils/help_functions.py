import os
import shutil
from glob import glob
import json
import pandas as pd
import numpy as np
import  cv2


def show_sample_with_labels(img_path, bbox_list, img_height, img_width):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for box in bbox_list:
        box = box.split()
        x1, y1 = float(box[1]) * img_width, float(box[2]) * img_height
        x2, y2 = (float(box[1]) + float(box[3])) * img_width, (float(box[2]) + float(box[4])) * img_height
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imwrite("img.jpg", img)

def divide_to_sets_by_area(data_folder_path):
    def assign_bin(query, bin_edges):
        for idx, bin_edge in enumerate(bin_edges):
            if idx == bin_edges.shape[0] - 1:
                if query == bin_edge:
                    return idx
                else:
                    return np.nan
            elif bin_edge <= query < bin_edges[idx+1]:
                return idx
        return np.nan

    if not os.path.isdir(os.path.join(data_folder_path, 'images', 'train')):
        os.makedirs(os.path.join(data_folder_path, 'images', 'train'))
    if not os.path.isdir(os.path.join(data_folder_path, 'images', 'val')):
        os.makedirs(os.path.join(data_folder_path, 'images', 'val'))

    with open(os.path.join(data_folder_path, 'labels.json')) as f:
        j = json.load(f)
    labels = pd.DataFrame(j['annotations'])
    images = pd.DataFrame(j['images'])

    data_by_area = []
    for id, group in labels.groupby('id'):
        data_by_area.append({'id': id, 'mean_area': group['area'].mean()})

    bins, bin_edges = np.histogram(pd.DataFrame(data_by_area)['mean_area'])
    for img_data in data_by_area:
        img_data['area_bin'] = assign_bin(query=img_data['mean_area'], bin_edges=bin_edges)

    train_data, val_data, test_data = [], [], []
    df_name_area = pd.DataFrame(data_by_area)

    for bin in range(bins.shape[0]):
        if bins[bin] < 10:
            test_data += list(df_name_area.loc[df_name_area['area_bin'] == bin].T.to_dict().values())
        else:
            matching_data = list(df_name_area.loc[df_name_area['area_bin'] == bin].T.to_dict().values())
            train_data += matching_data[:int(0.6*len(matching_data))]
            val_data += matching_data[int(0.6*len(matching_data)):int(0.8*len(matching_data))]
            test_data += matching_data[int(0.8*len(matching_data)):]

    train_df = pd.DataFrame(train_data)['id'].astype(np.int64).apply(lambda x: '{:04}'.format(x)).to_frame()
    val_df = pd.DataFrame(val_data)['id'].astype(np.int64).apply(lambda x: '{:04}'.format(x)).to_frame()
    test_df = pd.DataFrame(test_data)['id'].astype(np.int64).apply(lambda x: '{:04}'.format(x)).to_frame()

    train_df.columns = ["img_id"]
    val_df.columns = ["img_id"]
    test_df.columns = ["img_id"]

    train_df.to_csv(os.path.join(data_folder_path, 'train_ids.csv'), index=False)
    val_df.to_csv(os.path.join(data_folder_path, 'val_ids.csv'), index=False)
    test_df.to_csv(os.path.join(data_folder_path, 'test_ids.csv'), index=False)


if __name__ == '__main__':
    # divide_to_sets_by_area(data_folder_path='/home/amichay/DL/RetailPricingDetection/data')
    with open('/home/amichay/DL/RetailPricingDetection/data/labels/0001.txt', 'r') as f:
        labels = f.read().strip().splitlines()
    show_sample_with_labels(img_path='/home/amichay/DL/RetailPricingDetection/data/images/0001.jpg',
                            bbox_list=labels,
                            img_height=1836,
                            img_width=3264)




