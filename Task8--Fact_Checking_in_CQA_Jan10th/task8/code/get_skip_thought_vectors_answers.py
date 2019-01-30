import os
import read_data_file as rdf
from skipthoughts import skipthoughts as skt
import pandas as pd
import numpy as np
import pickle

curr_dir = os.path.dirname(os.path.abspath(__file__))
folder = "train_and_dev_sets_questions_and_an"


def save_vectors_for_texts(answer_ids, answer_subjects, answer_bodies, answer_answers, type_of_data,
                                                    combination_type="concatenation", without_urls = False):
    """
    DESCRIPTION:
        Generate skipthought vectors and save to numpy array files - X_data_bodies, X_data_subjects,
        X_data_combined_concat and X_data_combined_average as per given parameters.

    PARAMETERS:
        combined_type: Takes values 'concatenation' and 'average'
            - concatenation : concatenates both body and subject of a particular question by " " and
              then generate skipthought vector.
            - average : calculate skipthought vector for body, subject and answer separately and take mean of three.
        type_of_data : Possible values - train, dev
        without_urls : Removes URLs from text if it is True. Otherwise, URLs remain in text

    RETURNS:
        Skipthought vectors that are generated.

    """

    skipthought_vectors = []
    model = skt.load_model()
    encoder = skt.Encoder(model)
    print encoder

    filename_of_data = ""
    if without_urls:
        filename_of_data = "without_urls_" + filename_of_data

    if combination_type == "concatenation":
        combined_text = []
        for i, txt in enumerate(answer_subjects):
            if answer_bodies[i] is not None:
                combined = str(u' '.join([txt, answer_bodies[i], answer_answers[i]]).encode('ascii', 'ignore').decode('ascii').strip())
            else:
                combined = str(u' '.join([txt, answer_answers[i]]).encode('ascii', 'ignore').decode('ascii').strip())

            combined_text.append(combined)

        print "succeded"

        skipthought_vectors = np.asarray(encoder.encode(combined_text))
        filename_of_data += "answers_X_data_combined_concat"

    if combination_type == "average":
        body_list = [] # contains question bodies list
        subj_list = [] # contains question subjects list
        answ_list = [] # contains answers list

        for ind, body in enumerate(answer_bodies):
            # print body
            # print answer_ids[ind]
            if body is None: # Some samples have body as None. Taking subject value for body for such samples.
                body_list.append(str(answer_subjects[ind].encode('ascii', 'ignore').decode('ascii').strip()))
            else:
                body_list.append(str(body.encode('ascii', 'ignore').decode('ascii').strip()))
            subj_list.append(str(answer_subjects[ind].encode('ascii', 'ignore').decode('ascii').strip()))
            answ_list.append(str(answer_answers[ind].encode('ascii', 'ignore').decode('ascii').strip()))

        body_stv = np.asarray(encoder.encode(body_list))
        subj_stv = np.asarray(encoder.encode(subj_list))
        answ_stv = np.asarray(encoder.encode(answ_list))

        skipthought_vectors = (body_stv + subj_stv + answ_stv) / 3.0
        filename_of_data += "answers_X_data_combined_average"

    data_dict = {"ids" : answer_ids, "data" : skipthought_vectors}

    filename_of_data += "_" + type_of_data
    print "Saving data to " + filename_of_data + ".pickle file ....!"
    with open(filename_of_data+'.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print "Saved...!"

    return data_dict


def write_data_to_pickle(filename, file_type, combination_type="concatenation", without_urls = False):

    file_path = os.path.join(curr_dir, '..', folder, filename)

    answers_dict = rdf.read_answer_labels_from_xml(file_path, without_urls)
    print("No of " + file_type + " samples :", len(answers_dict))
    # print answers_dict

    file_to_save = "answers_dict_" + file_type
    if without_urls:
        file_to_save = "without_urls_" + file_to_save

    # saving to pickle file
    print("Saving data to answers_dict_" + file_type + ".pickle file ....!")
    with open(file_to_save + ".pickle", 'wb') as handle:
        pickle.dump(answers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved...!")

    answer_ids, answer_bodies, answer_subjects, answer_answers = rdf.get_lists_from_dict_answers(answers_dict)

    X_data = save_vectors_for_texts(answer_ids, answer_subjects, answer_bodies, answer_answers, file_type,
                                    combination_type, without_urls = without_urls)
    print X_data['data'].shape
    print len(X_data['ids'])



# file = "answers_train.xml"
# write_data_to_pickle(file, "train", combination_type="average") # Writing training Data
#
# file = "answers_dev.xml"
# write_data_to_pickle(file, "dev", combination_type="average") # Writing dev Data

file = "answers_test.xml"
write_data_to_pickle(file, "test", combination_type="concatenation") # Writing dev Data
write_data_to_pickle(file, "test", combination_type="average") # Writing dev Data










# print "Writing questions dictionary to questions_dict.pickle file ....!"
# with open('questions_dict.pickle', 'wb') as handle:
#     pickle.dump(questions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print "Written...!"
# exit(0)



# model = skt.load_model()
# encoder = skt.Encoder(model)
# print  "Processing on random data..!"
# data = ["Hello this is adithya", ".", "hi, I am gud."]
# qids = ["A", "B", 'C']
# vec = encoder.encode(data)
# data_dict = {"ids" : qids, "data" : data}
# print vec[1]
# print np.asarray(vec).shape
#
# import pickle
#
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('filename.pickle', 'rb') as handle:
#     unserialized_data = pickle.load(handle)
#
# print(data_dict == unserialized_data)


# print len(question_ids)
# print len(question_subjects)
# print len(question_bodies)
# All lengths are 1118

# In [1]: import numpy as np
#
# In [2]: old_set = [[0, 1], [4, 5]]
#
# In [3]: new_set = [[2, 7], [0, 2]]
#
# In [4]: (np.array(old_set) + np.array(new_set)) / 2
# Out[4]:
# array([[1. , 4. ],
#        [2. , 3.5]])
#
# In [5]: np.mean( np.array([ old_set, new_set ]), axis=0 )
# Out[5]:
# array([[1. , 4. ],
#        [2. , 3.5]])

#pandas_df = pd.DataFrame(data=questions_dict)
#print pandas_df
# None qid is Q196_R27


# print questions_dict
# print(questions_dict['Q317_R16'])

# print encoder.encode("questions_dict['Q1_R6']['body']")
# print len(encoder.encode("questions_dict['Q1_R6']['body']"))
#
# print encoder.encode("questions_dict['Q1_R6']['subject']")
# print len(encoder.encode("questions_dict['Q1_R6']['subject']"))


'''
for qid in questions_dict:
    if questions_dict[qid]['label'] == 1:
        if questions_dict[qid]['body'] is None:
            print questions_dict[qid]['subject']
        elif questions_dict[qid]['subject'] is None:
            print questions_dict[qid]['body']
        else:
            print questions_dict[qid]['body'] + " ---- " + questions_dict[qid]['subject']
'''