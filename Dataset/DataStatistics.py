import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import pandas as pd


def bboxes_area(annotations_list):
    annot_df = pd.DataFrame(annotations_list)
    areas = annot_df['area'].to_numpy()
    fig, ax = plt.subplots()
    n, bins, _ = ax.hist(x=areas)
    ax.set_title('Histogram for bboxes area')
    ax.set_xlabel('area')
    ax.set_ylabel('P')


if __name__ == '__main__':
    with open('/home/amichay/DL/RetailPricingDetection/data/labels.json') as json_file:
        data = json.load(json_file)
    bboxes_area(annotations_list=data['annotations'])