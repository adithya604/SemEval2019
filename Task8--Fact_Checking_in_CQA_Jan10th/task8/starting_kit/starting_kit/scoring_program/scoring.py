from read_data import read_answer_labels_from_xml, read_answer_predictions, read_question_labels_from_xml, read_question_predictions

CLASSES = [0, 1, 2]

def evaluate_answer_predictions(gold_labels_file, predictions_file, map_at=20):
    gold_labels = read_answer_labels_from_xml(gold_labels_file)
    predictions = read_answer_predictions(predictions_file)

    accuracy, f1, avgrec = calculate_metrics(gold_labels, predictions) 
    mean_average_precision = calculate_map(map_at, gold_labels, predictions)

    return accuracy, f1, avgrec, mean_average_precision


def evaluate_question_predictions(gold_labels_file, predictions_file):
    gold_labels = read_question_labels_from_xml(gold_labels_file)
    predictions = read_question_predictions(predictions_file)

    accuracy, f1, avgrec = calculate_metrics(gold_labels, predictions) 

    return accuracy, f1, avgrec


def calculate_metrics(gold_labels, predicted_labels):
    metrics = {}
    # calculate metrics for each class
    print "GOLD LABELS", gold_labels.keys()
    print "PREDICTED LABELS", predicted_labels.keys()
    print len(gold_labels.keys())
    print len(predicted_labels.keys())


    for target_class in CLASSES:
        acc, f1, avgrec = metrics_per_class(gold_labels, predicted_labels, target_class)
        metrics[target_class] = [acc, f1, avgrec]

    # Return the macro metrics from the metrics in each class
    num_classes = len(CLASSES)

    accuracy = sum([float(m[0]) for m in metrics.values()]) / num_classes
    f1 = sum([float(m[1]) for m in metrics.values()]) / num_classes
    avgrec = sum([float(m[2]) for m in metrics.values()]) / num_classes

    return accuracy, f1, avgrec


def metrics_per_class(gold_labels, predicted_labels, target_class):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for object_id, predictions in predicted_labels.items():
        gold_label = gold_labels[object_id]
        predicted_label = predictions[0] # the choice of this way to read the label is because need flexibility to use this function for both questions and answers
        # print(object_id, predicted_label, gold_label)
        if gold_label == target_class and predicted_label == target_class:
            true_positives += 1
        elif gold_label == target_class and predicted_label != target_class:
            false_negatives += 1
        elif gold_label != target_class and predicted_label != target_class:
            true_negatives += 1
        elif gold_label != target_class and predicted_label == target_class:
            false_positives += 1

    confusion_matrix = 'TP=' + str(true_positives) + ', FP=' + str(false_positives) + ', TN=' + str(true_negatives) + ', FN=' + str(false_negatives) 
    confusion_matrix2 = 'PredictedYES='+str((true_positives+false_positives))+', ActualYES='+str((true_positives+false_negatives))
    confusion_matrix3 = 'PredictedNO='+str((true_negatives+false_negatives))+', ActualNO='+str((true_negatives+false_positives))
    print(confusion_matrix, '\n', confusion_matrix2, '\n', confusion_matrix3)

    accuracy = float((true_positives + true_negatives))/(true_positives+true_negatives+false_positives+false_negatives)

    if true_positives+false_positives > 0:
        precision = float(true_positives)/(true_positives+false_positives)
    else:
        precision = 0

    recall = float(true_positives)/(true_positives+false_negatives)

    f1 = 2*float(true_positives)/(2*true_positives + false_negatives + false_positives)

    return accuracy, f1, recall


def calculate_map(p, gold_labels, predicted_scores):
    map_value = 0
    scores = {}
    counter = 0
    # Put all values in a map for each query
    for answer_id, [_, predicted_score] in predicted_scores.items():
        qid = qid_from_cid(answer_id)
        gold_label = gold_labels[answer_id]
        add_ranking = 1./cid_to_int_extracted(answer_id)*0.0000000001
        if not qid in scores.keys():
            scores[qid] = {}
        scores[qid][predicted_score+add_ranking] = gold_label

    for query, score_label_mapping in scores.items():
        # print(score_label_mapping.values())
        if 1 in score_label_mapping.values() and sum(score_label_mapping.values()) < len(score_label_mapping.values()):
            counter += 1
            # print(query, score_label_mapping)
            sorted_scores = sorted(score_label_mapping.keys(), reverse=True)
            average_precision = 0
            limit = min(p, len(sorted_scores))

            count_positive_labels = 0
            for i in range(0,limit):
                score = sorted_scores[i]
                label = score_label_mapping[score]
                if label == 1:
                    count_positive_labels += 1
                    average_precision += float(count_positive_labels)/(i+1)
            map_value += float(average_precision) / count_positive_labels
    map_value /= float(counter)

    return map_value



############## HELPERS START ###################

# Functions for extracting question and answer numeric values
# from given question or answer IDs, such as Q123_R45 or Q123_R45_C6
# The int repsresentation is used for easier sorting in MAP calculation.
# The functions assume the input values are in the correct format.

# Q123_R45 -> 12345000
def qid_to_int(qid):
    part1q = qid[qid.find('Q')+1:qid.find('_R')]
    part2q = qid[qid.find('_R')+2:]
    resq = int(part1q)*10000000 + int(part2q)*1000
    return resq

# Q123_R45_C6 -> Q123_R45
def qid_from_cid(cid):
    return cid[cid.find('Q'):cid.find('_C')]

# Q123_R45_C6 -> 12345006
def cid_to_int(cid):
    part1c = cid[cid.find('Q')+1:cid.find('_R')]
    part2c = cid[cid.find('_R')+2:cid.find('_C')]
    part3c = cid[cid.find('_C')+2:]
    resc = int(part1c)*10000000 + int(part2c)*1000 + int(part3c)
    return resc

# Q123_R45_C6 -> 6
def cid_to_int_extracted(cid):
    return int(cid[cid.find('_C')+2:])

############## HELPERS END ###################



