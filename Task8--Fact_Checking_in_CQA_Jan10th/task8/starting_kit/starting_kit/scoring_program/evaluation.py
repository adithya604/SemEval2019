#!/usr/bin/env python
import logging
import sys
import os
import os.path
from scoring import evaluate_answer_predictions, evaluate_question_predictions

# This file contains the script for scoring submitted question and answer predictions.
# 
# The same script will be used on evaluating the Codalab submissions, therefore
# the expected file structure is according to the Codalab documentation:
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
#
# input/ -> input directory
#   res/ -> directory containing the system predictions
#       predict_questions.txt -> file containing the question predictions in the format QID\tPREDICTED_LABEL
#       predict_answers.txt   -> file containing the answer predictions in the format CID\tPREDICTED_LABEL\tPREDICTED_SCORE
#   ref/ -> directory containing the reference data
#       input_questions.xml   -> input XML file contatining the questions on the reference set with gold labels
#       input_answers.xml     -> input XML file contatining the answers on the reference set with gold labels
# output/ -> output directory
#   scores.txt -> the result from the execution will be written in this file
# 
# The following script will be executed
# python evaluation.py <input_folder> <output_folder>

# as per the metadata file, input and output directories are the arguments


# [_, input_dir, output_dir] = sys.argv
input_dir = "../input"
output_dir = "../output"
submission_dir = os.path.join(input_dir, 'res')
reference_dir = os.path.join(input_dir, 'ref')

# The scoring result will be kept in this variable
result = ''
# At the beginning we assume that no submission file is found. 
# The value will be updated if a submission for either questions or answers is found
submission_found = False

# Read question predictions
submission_path_questions = os.path.join(submission_dir, 'predict_questions_first.txt')
# If the question predictions file was supplied, run the scoring on it
if os.path.exists(submission_path_questions):
    submission_found = True
    input_file_questions = os.path.join(reference_dir, 'input_questions.xml')
    # Write the questions predictions to the result
    acc, fscore, avgrec = evaluate_question_predictions(input_file_questions, submission_path_questions)
    result += "questions_accuracy:{0}\nquestions_f1:{1}\nquestions_avgrec:{2}\n".format(acc, fscore, avgrec)


# Read answer predictions
submission_path_answers = os.path.join(submission_dir, 'predict_answers.txt')
if os.path.exists(submission_path_answers):
    submission_found = True
    input_file_answers = os.path.join(reference_dir, 'input_answers.xml')
    # Evaluate the answers predictions to the result
    acc, fscore, avgrec, map_value = evaluate_answer_predictions(input_file_answers, submission_path_answers)
    result += "answers_accuracy:{0}\nanswers_fscore:{1}\nanswers_avgrec:{2}\nanswers_map:{3}\n".\
                        format(acc, fscore, avgrec, map_value)

# If no submission file was found, exit
if not submission_found:
    message = "Expected at least one submission file, found files {1}".\
                        format(os.listdir(submission_dir))
    logging.error(message)
    print(message)
    sys.exit(message)

print('Result:')
print(result)

# After all the data is processed, write the result to the output
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write(result)


