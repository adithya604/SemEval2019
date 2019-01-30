import xml.etree.ElementTree as ET
import csv, random, os
import operator
import io
import re

# This file contains functions for reading the data from input and prediction files.

ANSWER_LABELS_MAPPING = {'False': 0, 'True': 1, 'NonFactual': 2}
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}

# Reads answer labels from file in the task input format.
def read_question_labels_from_xml(input_xml_file, without_urls = False):
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
        if without_urls:
            each_ques_dict["subject"]   = string_parser(question_tag[0].text)
            if isinstance(question_tag[1].text, str):
                each_ques_dict["body"]  = string_parser(question_tag[1].text)
            else:
                each_ques_dict["body"]  = question_tag[1].text
        else:
            each_ques_dict["subject"] = question_tag[0].text
            each_ques_dict["body"]    = question_tag[1].text

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


# Get data in lists - to pass lists to SkipThoughts encoder
def get_lists_from_dict_answers(answers_dict, include_subject = True):
    answer_ids = []
    answer_bodies = []
    answer_subjects = []
    answer_answers = []

    for aid in answers_dict:
        answer_ids.append(aid)
        answer_bodies.append(answers_dict[aid]['body'])
        answer_subjects.append(answers_dict[aid]['subject'])
        answer_answers.append(answers_dict[aid]['answer'])

    return answer_ids, answer_bodies, answer_subjects, answer_answers


# Reads answer labels from file in the task input format.
def read_answer_labels_from_xml(input_xml_file, without_urls = False):
    print('parsing...', input_xml_file)

    answers_dict = {}

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    for thread in root:
        question_tag = thread[0]
        question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
        question_id = question_tag.attrib['RELQ_ID']

        if question_fact_label == 'Factual':
            for index, answer_tag in enumerate(thread):
                if index > 0: # the 0 index was processed above - it is the question
                    answer_fact_label = answer_tag.attrib['RELC_FACT_LABEL']
                    answer_id = answer_tag.attrib['RELC_ID']
                    label = get_label(answer_fact_label, ANSWER_LABELS_MAPPING)

                    each_answ_dict = {}
                    each_answ_dict["question_id"] = question_id
                    each_answ_dict["subject"] = question_tag[0].text
                    each_answ_dict["body"] = question_tag[1].text
                    each_answ_dict["answer"] = answer_tag[0].text
                    if label > -1:
                        each_answ_dict["label"] = label

                    answers_dict[answer_id] = each_answ_dict

    return answers_dict



def get_label(original_label, label_mapping):
    if original_label in label_mapping.keys():
        return label_mapping[original_label]

    return -1

def string_parser(text):
    # for removing the URLs from the given text
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return text