#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2019/7/9 13:54 
 @Author : ZHANG 
 @File : eval.py 
 @Description:
"""
import json
from tqdm import tqdm
from pprint import PrettyPrinter
from calculate_mAP import *

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(annotation, submit):
    """
    Evaluate.
    """
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    # read annotation file
    with open(annotation, 'r') as load_a:
        anno = json.load(load_a)

    for i, (_, bbox) in enumerate(anno.items()):
        box = list()
        box.append(bbox[i]["bbox"])
        label = list()
        label.append(bbox[i]["category_id"])
        for j in range (1, len(bbox)):
            if bbox[j]["image_id"] == bbox[j-1]["image_id"]:
                box.append(bbox[j]["bbox"])
                label.append(bbox[j]["category_id"])
            else:
                box = torch.FloatTensor(box)
                #box = [b.to(device) for b in box]
                label = torch.LongTensor(label)
                #label = [l.to(device) for l in label]
                true_boxes.append(box)
                box = list()
                box.append(bbox[j]["bbox"])
                true_labels.append(label)
                label = list()
                label.append(bbox[j]["category_id"])
        box = torch.FloatTensor(box)
        #box = [b.to(device) for b in box]
        label = torch.LongTensor(label)
        #label = [l.to(device) for l in label]
        true_boxes.append(box)
        true_labels.append(label)

    # read submit file
    with open(submit, 'r') as load_a:
        sub = json.load(load_a)

    for i, (_, bbox) in enumerate(sub.items()):
        box = list()
        box.append(bbox[i]["bbox"])
        label = list()
        label.append(bbox[i]["category_id"])
        score = list()
        score.append(bbox[i]["score"])
        for j in range (1, len(bbox)):
            if bbox[j]["image_id"] == bbox[j-1]["image_id"]:
                box.append(bbox[j]["bbox"])
                label.append(bbox[j]["category_id"])
                score.append(bbox[j]["score"])
            else:
                box = torch.FloatTensor(box)
                #box = [b.to(device) for b in box]
                label = torch.LongTensor(label)
                #label = [l.to(device) for l in label]
                score = torch.FloatTensor(score)
                det_boxes.append(box)
                box = list()
                box.append(bbox[j]["bbox"])
                det_labels.append(label)
                label = list()
                label.append(bbox[j]["category_id"])
                det_scores.append(score)
                score = list()
                score.append(bbox[j]["score"])
        box = torch.FloatTensor(box)
        #box = [b.to(device) for b in box]
        label = torch.LongTensor(label)
        #label = [l.to(device) for l in label]
        score = torch.FloatTensor(score)
        det_boxes.append(box)
        det_labels.append(label)
        det_scores.append(score)

    # Calculate mAP
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

    # Print AP for each class
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
    return mAP