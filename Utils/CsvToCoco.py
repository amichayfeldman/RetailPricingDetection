import pandas as pd
import json
import os
import numpy as np
import cv2


def annotations_to_yolo(df, images_folder, output_path):
    data = {}
    for idx, row in df.iterrows():
        file_name = row['img_name'].split('.')[0]
        img = cv2.imread(os.path.join(images_folder, f"{file_name}.jpg"))
        height, width = 640, 640
        box_width = (float(row['x2']) - float(row['x1'])) / width
        box_height = (float(row['y2']) - float(row['y1'])) / height
        x1 = row['x1'] / width
        y1 = row['y1'] / height

        if file_name not in data:
            data[file_name] = [[0, x1, y1, box_width, box_height]]
        else:
            data[file_name].append([0, x1, y1, box_width, box_height])

    for key, bbox_list in data.items():
        with open(os.path.join(output_path, f"{key}.txt"), "w") as f:
            for box in bbox_list:
                f.write(' '.join(str(i) for i in box))
                f.write('\n')
        f.close()


def parse_csv_to_dict(csv_path, images_folder):
    out_data = {'images': [], 'annotations': []}
    csv_data = pd.read_csv(csv_path)
    img_id_history = []
    for idx, row in csv_data.iterrows():
        img_id = int(row['img_name'].split('.')[0])
        file_name = row['img_name']
        img = cv2.imread(os.path.join(images_folder, file_name))
        height, width = img.shape[:2]
        if img_id not in img_id_history:
            img_id_history.append(img_id)
            out_data['images'].append({'id': img_id,
                                       'license': 1,
                                       'file_name': f"{file_name}",
                                       "height": height,
                                       "width": width,
                                       "date_captured": "null"})
        x, y = row['x1'], row['y1']
        h, w = row['y2'] - row['y1'], row['x2'] - row['x1']
        out_data['annotations'].append({"id": img_id,
                                        "image_id": img_id,
                                        "category_id": 0,
                                        'bbox': [x, y, w, h],
                                        'segmentation': [],
                                        'area': float(w*h),
                                        'iscrowd': 0})
    return out_data


def export_labels_yolo_format(images_df, labels_df, labels_folder):
    if not os.path.isdir(labels_folder):
        os.makedirs(labels_folder)
    for idx, row in images_df.iterrows():
        file_name = row['file_name'].split('.')[0]
        label_file_path = os.path.join(labels_folder, f"{file_name}.txt")
        matching_labels = labels_df.loc[labels_df['id'] == row['id']]
        with open(label_file_path, 'a') as f:
            for jdx, label_row in matching_labels.iterrows():
                line = f"0 {label_row['bbox'][0] / row['width']} {label_row['bbox'][1] / row['height']} " \
                       f"{label_row['bbox'][2] / row['width']} {label_row['bbox'][3] / row['height']}\n"
                f.write(line)
        f.close()


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
    yolo_format_labels = '/home/amichay/DL/RetailPricingDetection/data/labels'

    annotations_to_yolo(df = pd.read_csv('/home/amichay/DL/RetailPricingDetection/data/annotations.csv'),
                        images_folder=images_folder,
                        output_path='/home/amichay/DL/RetailPricingDetection/data/labels')


    #
    # data = parse_csv_to_dict(csv_path=csv_path, images_folder=images_folder)
    # write_info_to_json(json_path=json_out_path, data_dict=data)
    # export_labels_yolo_format(images_df=pd.DataFrame(data['images']), labels_df=pd.DataFrame(data['annotations']),
    #                           labels_folder=yolo_format_labels)
    #

    import pandas as pd
    import os
    import shutil

    train_data = pd.read_csv('/home/amichay/DL/RetailPricingDetection/data/train_ids.csv')
    val_data = pd.read_csv('/home/amichay/DL/RetailPricingDetection/data/val_ids.csv')

    for idx, row in train_data.iterrows():
        img_path = os.path.join('/home/amichay/DL/RetailPricingDetection/data/images', "{:04}.jpg".format(row['img_id']))
        target_path = os.path.join('/home/amichay/DL/RetailPricingDetection/data/images/train2', "{:04}.jpg".format(row['img_id']))
        shutil.copy2(src=img_path, dst=target_path)

        label_path = os.path.join('/home/amichay/DL/RetailPricingDetection/data/labels', "{:04}.txt".format(row['img_id']))
        label_target_path = os.path.join('/home/amichay/DL/RetailPricingDetection/data/labels/train2', "{:04}.txt".format(row['img_id']))
        shutil.copy2(src=label_path, dst=label_target_path)