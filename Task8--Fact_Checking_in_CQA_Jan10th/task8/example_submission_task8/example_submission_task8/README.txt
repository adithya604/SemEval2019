This folder contains an example submission for the SemEval 2019 Task 8 on Fack-Checking in Community Question Answering.


## Example

Please take a look at the submission example in this directory. 

The file *baseline_submission.zip* contains the files with required naming and formatting convention for a successful submission to the competition.

They contain the majority class label from the corresponding training set.


Please read below for submission details.


## Submission Content

The submission contains one file for each subtask. If you decide to only participate in one subtask, you can only include the file for that subtask. 

* The file containing your result for "Subtask A - Question Classification", must have the name *predict_questions.txt*.

* The file containing your result for "Subtask B - Answer Classification", must have the name *predict_answers.txt*.

* Both files need to be put in a ZIP archive, *without an additional directory*.


** Submission to both subtasks: Please keep in mind the following:**

If you want to participate in both subtasks, but you want to only update the submission for one of the subtasks, you should submit the files for *both* subtasks. 

If you upload a submission which contains only one file, the leaderboard will contain only this submission and your old submission for the other subtask will disappear from the leaderboard. 

This is required because of the Codalab functionality for handling multiple leaderboards as part of one competition and cannot be avoided.


## Submission File Format

### Submission File Format for Subtask A - Question Classification:

The file *predict_questions.txt* should contain one line per predicted question. Each line should have the following format:

'''
RELQ_ID\tLABEL
'''

where:

* RELQ_ID is the value of the question ID (field RELQ_ID from the XML file thread)

* LABEL is the predicted label, presented as a numerical value as follows:

0 - Opinion
1 - Factual
2 - Socializing

**For example, if you want to have a line which predicts label "Factual" with ID Q1_R1, then the line will have the following format '''Q1_R1\t1'''.**


### Submission File Format for Subtask B - Answer Classification:

The file *predict_questions.txt* should contain one line per predicted question. Each line should have the following format:

'''
RELQ_ID\tLABEL\tFACT_SCORE
'''

where:

* RELQ_ID is the value of the question ID (field RELQ_ID from the XML file thread)

* LABEL is the predicted label, presented as a numerical value as follows:

0 - False
1 - True
2 - NonFactual

* FACT_SCORE is a float value representing the score for the factual label. The idea of this field is to evaluate the system also in terms of Mean Average Precision (MAP), i.e. evaluating how well the system can order the factual true answers "above" the false or non-factual ones.


**For example, if you want to have a line which predicts label "True" and score for the factual label 0.8 for answer with ID Q1_R1_C1, then the line will have the following format '''Q1_R1\t1\t0.8'''.**



If you have any questions or problems, please contact the task organizers on the following email address: semeval-2019-task-8-organizers@googlegroups.com 


Good luck!


