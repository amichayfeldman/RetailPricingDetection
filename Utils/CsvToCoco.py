import pandas as pd
import json
import os
import numpy as np
import cv2


def parse_csv_to_dict(csv_path, images_folder):
    out_data = {'images': [], 'annotations': []}
    csv_data = pd.read_csv(csv_path)
    for idx, row in csv_data.iterrows():
        file_name = row['img_name']
        img = cv2.imread(os.path.join(images_folder, file_name))
        height, width = img.shape[:2]
        out_data['images'].append({'id': idx,
                                   'license': 1,
                                   'file_name': f"{file_name}",
                                   "height": height,
                                   "width": width,
                                   "date_captured": "null"})
        x, y = row['x1'], row['y1']
        h, w = row['y2'] - row['y1'], row['x2'] - row['x1']
        out_data['annotations'].append({"id": idx,
                                        "image_id": idx,
                                        "category_id": 1,
                                        'bbox': [x, y, w, h],
                                        'segmentation': [],
                                        'area': float(w*h),
                                        'iscrowd': 0})
    return out_data


def write_info_to_json(json_path, data_dict):
    out_dict = {}
    out_dict['info'] = {'year': "2021",
                        "version": 1.0,
                        "description": "Retail pricing detection",
                        "contributor": "AmichayFeldman",
                        "url": "",
                        "date_created": "2021-04-06T20:38:00"}
    out_dict['licenses'] = [{"url": "",
                             "id": 1,
                             "name": "Normal"}]
    out_dict['categories'] = [{"id": 1,
                               "name": "bottle",
                               "supercategory": "bottle"}]
    out_dict['images'] = data_dict['images']
    out_dict['annotations'] = data_dict['annotations']

    with open(json_path, 'w') as fout:
        json.dump(out_dict, fout, indent=4)


if __name__ == '__main__':
    csv_path = '/home/amichay/DL/RetailPricingDetection/data/annotations.csv'
    images_folder = '/home/amichay/DL/RetailPricingDetection/data/images'
    json_out_path = '/home/amichay/DL/RetailPricingDetection/data/labels.json'

    data = parse_csv_to_dict(csv_path=csv_path, images_folder=images_folder)
    write_info_to_json(json_path=json_out_path, data_dict=data)