import xml.etree.ElementTree as ET
import csv, random, os
import operator
import io

# This file contains functions for reading the data from input and prediction files.

ANSWER_LABELS_MAPPING = {'False': 0, 'True': 1, 'NonFactual': 2}
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}

# Reads answer labels from file in the task input format.
def read_question_labels_from_xml(input_xml_file):
    questions_dict = {}
    print('parsing...', input_xml_file)

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_id = question_tag.attrib['RELQ_ID']
        question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
        label = get_label(question_fact_label, QUESTION_LABELS_MAPPING)

        each_ques_dict = {}
        if label > -1:
            each_ques_dict["label"] = label
        each_ques_dict["subject"]   = question_tag[0].text
        each_ques_dict["body"]      = question_tag[1].text
        questions_dict[question_id] = each_ques_dict
        """
        print(thread)
        print(len(thread))
        print(question_tag)
        print(question_id)
        print(question_fact_label)
        print(question_tag[0].text)
        print(question_tag[1].text)
        """
    return questions_dict

# Get data in lists - to pass lists to SkipThoughts encoder
def get_lists_from_dict(questions_dict, include_subject = True):
    # question_ids = []
    # question_bodies = []
    # question_subjects = []
    #
    # for qid in questions_dict:
    #     question_ids.append(qid)
    #     if questions_dict[qid]['body'] is not None:
    #         question_bodies.append([questions_dict[qid]['body']])
    #     else:
    #         question_bodies.append("")
    #     if include_subject:
    #         question_subjects.append([questions_dict[qid]['subject']])
    # return question_ids, question_bodies, question_subjects

    question_ids = []
    question_bodies = []
    question_subjects = []

    for qid in questions_dict:
        if questions_dict[qid]['subject'] is not None:
            question_ids.append(qid)
            question_bodies.append([questions_dict[qid]['body']])
            if include_subject:
                question_subjects.append([questions_dict[qid]['subject']])

    return question_ids, question_bodies, question_subjects




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



def get_label(original_label, label_mapping):
    if original_label in label_mapping.keys():
        return label_mapping[original_label]

    return -1