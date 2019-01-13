# Starting kit for the SemEval 2019 Task 8 "Fact Checking in Community Question Answering Forums"

This is the starting kit for the SemEval 2019 Task 8 "Fact Checking in Community Question Answering Forums".

## Contents

This archive contains the following:

input/ -> input directory
    res/ -> directory containing the system predictions
        predict_questions.txt -> file containing the question predictions for the trial dataset 
                                    in the format QID\tPREDICTED_LABEL
        predict_answers.txt   -> file containing the answer predictions for the trial dataset 
                                    in the format CID\tPREDICTED_LABEL\tPREDICTED_SCORE
    ref/ -> directory containing the reference data
        input_questions.xml   -> input XML file contatining the questions on the trial dataset
        input_answers.xml     -> input XML file contatining the answers on the trial dataset

output/ -> output directory
    scores.txt -> the result from the execution will be written in this file

scoring_program/ -> contains scoring script.
    evaluation.py -> the script to be executed for scoring the predictions 
    read_data.py -> helper methods for reading data
    scoring.py -> the methods for computing scoring values


The same script in the 'scoring_peogram/' will be used on evaluating the Codalab submissions, 
therefore the expected file structure is according to the Codalab documentation:
https://github.com/codalab/codalab-competitions/wiki/User-Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions


## Score your predictions

In order to score your predictions:

1. Replace the files in directory *input/res* with the prediction files from your system.
The files must have the same format and be named the same way.

2. Execute the scoring script:

‘python scoring_program/evaluate.py input output‘


## Prepare Submission

Your submission for the Codalab website should contain two separate files for each subtask. The files must be named exactly the same as the files in the *input/res* directory - *predict_questions.txt* for the question predictions and *predict_answers.txt* for the answer predictions.

The submission is a ZIP file containing on the base level (in no folder) the two files, described above.

### Question Predictions

The question predictions file - *predict_questions.txt*, contains the question labels predicted by your system. 
Each line contains prediction for one question - question ID and predicted label, separated by a TAB symbol (\t).

The example file *input/res/predict_questions.txt* contains the correct structure.

### Answer Predictions

The question predictions file - *predict_answers.txt*, contains the answer labels predicted by your system. 
Each line contains prediction for one answer - answer ID, predicted label and predicted scoring, separated by a TAB symbol (\t).

The example file *input/res/predict_answers.txt* contains the correct structure.







