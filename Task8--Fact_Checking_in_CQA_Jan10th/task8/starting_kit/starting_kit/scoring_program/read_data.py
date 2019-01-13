import xml.etree.ElementTree as ET
import csv, random, os
import operator
import io

# This file contains functions for reading the data from input and prediction files.

ANSWER_LABELS_MAPPING = {'False': 0, 'True': 1, 'NonFactual': 2}
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}


# Reads answer labels from file in the task input format.
def read_answer_labels_from_xml(input_xml_file):
    labels = {}
    print('parsing...', input_xml_file)

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
        if question_fact_label == 'Factual':
            for index, answer_tag in enumerate(thread):
                if index > 0: # the 0 index was processed above - it is the question
                    answer_fact_label = answer_tag.attrib['RELC_FACT_LABEL']
                    answer_id = answer_tag.attrib['RELC_ID']
                    label = get_label(answer_fact_label, ANSWER_LABELS_MAPPING)
                    if label > -1:
                        labels[answer_id] = label
    return labels

# Reads answer labels from file in the task input format.
def read_question_labels_from_xml(input_xml_file):
    labels = {}
    print('parsing...', input_xml_file)

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_id = question_tag.attrib['RELQ_ID']
        question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
        label = get_label(question_fact_label, QUESTION_LABELS_MAPPING)
        if label > -1:
            labels[question_id] = label

    return labels

# Reads answer labels and scores from TSV file in the format: CID\tlabel\tscore
# Returns map: {CID: [pred_label, pred_score]}
def read_answer_predictions(predictions_file):
    lines = io.open(predictions_file).read().strip().split('\n')
    predictions = {}
    for line in lines:
        cid, pred_label, pred_score = line.split('\t')
        predictions[cid] = [int(pred_label), float(pred_score)]
    return predictions

# Reads answer labels and scores from TSV file in the format: CID\tlabel\tscore
# Returns map: {CID: [pred_label, pred_score]}
def read_question_predictions(predictions_file):
    lines = io.open(predictions_file).read().strip().split('\n')
    predictions = {}
    for line in lines:
        qid, pred_label = line.split('\t')
        predictions[qid] = [int(pred_label)]
    return predictions

def get_label(original_label, label_mapping):
    if original_label in label_mapping.keys():
        return label_mapping[original_label]

    return -1

