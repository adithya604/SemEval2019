import os
import read_data_file as rdf
from skipthoughts import skipthoughts as skt
import pandas as pd
import numpy as np
import pickle

curr_dir = os.path.dirname(os.path.abspath(__file__))
folder = "train_and_dev_sets_questions_and_an"


def save_vectors_for_texts(question_ids, question_subjects, question_bodies, type_of_data, vector_for="body",
                                                    combination_type="concatenation"):
    """
    DESCRIPTION:
        Generate skipthought vectors and save to numpy array files - X_data_bodies, X_data_subjects,
        X_data_combined_concat and X_data_combined_average as per given parameters.

    PARAMETERS:
        vector_for: Takes values 'body', 'subject', 'both'
            - body : returns skipthought vector considering only question bodies
            - subject :  skipthought vector considering only question subjects
            - both : returns skipthought vector based on parameter - combination_type
            Default Value : 'body'
        combined_type: Takes values 'concatenation' and 'average'
            - concatenation : concatenates both body and subject of a particular question by " " and
              then generate skipthought vector.
            - average : calculate skipthought vector for body and subject separately and take mean of both.
        type_of_data : Possible values - train, dev

    RETURNS:
        Skipthought vectors that are generated.

    """

    skipthought_vectors = []
    model = skt.load_model()
    encoder = skt.Encoder(model)
    print encoder

    filename_of_data = ""
    id_list = question_ids

    if vector_for == "body":
        body_list = [] # contains question bodies list without None Values in question body
        id_list = []
        for ind, body in enumerate(question_bodies):
            # print(body)
            if body[0] is not None:
                id_list.append(question_ids[ind])
                # print body
                # print len(body)
                body_list.append(str(body[0].encode('ascii', 'ignore').decode('ascii').strip()))

        skipthought_vectors = np.asarray(encoder.encode(body_list))
        filename_of_data = "X_data_bodies"

    elif vector_for == "subject":   # No None values in question subjects
        skipthought_vectors = np.asarray(encoder.encode(question_subjects))
        filename_of_data = "X_data_subjects"

    elif vector_for == "both" and combination_type == "concatenation":
        combined_text = []
        for i, txt in enumerate(question_subjects):
            if question_bodies[i][0] is not None:
                # print txt[0], "---", question_bodies[i][0]
                combined = str(u' '.join((txt[0], question_bodies[i][0])).encode('ascii', 'ignore').decode('ascii').strip())
                combined_text.append(combined)
            else:
                combined_text.append(str(txt[0].encode('ascii', 'ignore').decode('ascii').strip()))
        print "succeded"

        # exit(0)

        skipthought_vectors = np.asarray(encoder.encode(combined_text))
        filename_of_data = "X_data_combined_concat"

    elif vector_for == "both" and combination_type == "average":
        body_list = [] # contains question bodies list without None Values in question body
        subj_list = [] # contains question subjects list without None Values in question body
        id_list = []   # contains question ids list without None Values in question body
        for ind, body in enumerate(question_bodies):
            if body[0] is not None:
                id_list.append(question_ids[ind])
                body_list.append(str(body[0].encode('ascii', 'ignore').decode('ascii').strip()))
                subj_list.append(str(question_subjects[ind][0].encode('ascii', 'ignore').decode('ascii').strip()))
        body_stv    = np.asarray(encoder.encode(body_list))
        subject_stv = np.asarray(encoder.encode(subj_list))
        skipthought_vectors = (body_stv + subject_stv) / 2.0
        filename_of_data = "X_data_combined_average"

    data_dict = {"ids" : id_list, "data" : skipthought_vectors}

    # print "Saving vectors to " + filename_of_data + " ....!"
    # np.save(filename_of_data, arr=skipthought_vectors)
    # print "Saved...!"

    filename_of_data = filename_of_data + "_" + type_of_data
    print "Saving data to " + filename_of_data + ".pickle file ....!"
    with open(filename_of_data+'.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print "Saved...!"

    return data_dict


def write_data_to_pickle(filename, file_type, vector_for="body", combination_type="concatenation"):

    file_path = os.path.join(curr_dir, '..', folder, filename)

    questions_dict = rdf.read_question_labels_from_xml(file_path)
    print("No of " + file_type + " samples :", len(questions_dict))

    file_to_save = "questions_dict_" + file_type

    # Saving to numpy file
    print("Saving vectors to numpy file " + file_to_save + " ....!")
    np.save(file_to_save, arr=questions_dict)
    print("Saved...!")

    # # saving to pickle file
    print("Saving data to questions_dict_" + file_type + ".pickle file ....!")
    with open(file_to_save + ".pickle", 'wb') as handle:
        pickle.dump(questions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved...!")

    # print questions_dict['Q1_R6']['body'], questions_dict['Q1_R6']['subject']

    question_ids, question_bodies, question_subjects = rdf.get_lists_from_dict(questions_dict)
    print "No of question samples now:", len(question_ids), len(question_bodies), len(question_subjects)

    X_data = save_vectors_for_texts(question_ids, question_subjects, question_bodies, file_type, vector_for,
                                                                                                    combination_type)
    print X_data['data'].shape
    print len(X_data['ids'])



file = "questions_train.xml"
write_data_to_pickle(file, "train", vector_for="both", combination_type="concatenation") # Writing training Data

file = "questions_dev.xml"
write_data_to_pickle(file, "dev", vector_for="both", combination_type="concatenation") # Writing dev Data










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