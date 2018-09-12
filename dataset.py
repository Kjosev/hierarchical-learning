import os
import pickle
import numpy as np
# import imagenet_helpers as ih
from sklearn.model_selection import train_test_split

# def load_data(dataset='awa2', num_classes=50):
#     """
#     Loads the ResNet features for the images and class ids
#     Class ids correspond to indexes in 'classes.txt'
#     Returns:
#         X - image features 
#         y - image class ids [0 - num_classes]
#     """
#     # if dataset == 'imagenet':
#         # return ih.load_data(num_classes)

#     images_path = 'data/awa2/awa2_resnet.npy'
#     labels_path = 'data/awa2/awa2_labels.csv'

#     X = np.load(images_path)
#     y = np.loadtxt(labels_path)

#     valid_idxs = y < num_classes
#     X_pruned = X[valid_idxs]
#     y_pruned = y[valid_idxs]

#     return X_pruned, y_pruned


def load_data():
    data_path = 'data/awa2/features'

    mapping = get_mapping()

    X_parts = []
    y_parts = []
    # load feature matrices for all classes
    for p in os.listdir(data_path):
        full_p = os.path.join(data_path, p)

        curr_wind = p.split('.')[0]
        curr_id = mapping[curr_wind]
        curr_id = curr_id 

        X_part = np.load(full_p)
        num_samples = X_part.shape[0]
        y_part = np.ones((num_samples,1)) * curr_id

        X_parts.append(X_part)
        y_parts.append(y_part)

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    y = np.squeeze(y)
    return X, y

def get_mapping():
    mapping = {}
    with open('data/awa2/classes.txt', 'r') as in_f:
        for line in in_f:
            [id, class_name] = line.split()
            mapping[class_name] = int(id) - 1 # 0-based indexing
    return mapping

    

def split_data(X, y, val=0.1, test=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test + val)) 
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=(val / (test + val)))

    return X_train, X_val, X_test, y_train, y_val, y_test

def class_ids_to_names(id_batches, class_names):
    class_names = np.array(class_names)
    names = []
    for ids in id_batches:
        idsf = np.array(ids,dtype=int)
        curr_class_names = class_names[idsf]
        names.append(curr_class_names)
    return names

def get_batch_idxs(data_size, batch_size):
    """
    Returns list of indices grouped in arrays of size batch_size
    """
    indexes = np.arange(data_size)
    np.random.shuffle(indexes)

    idx = 0
    batch_idxs = []

    # print(indexes)
    while idx <= data_size:
        batch_idxs.append(indexes[idx:idx+batch_size])
        idx += batch_size

    return batch_idxs

def generate_hypernym_dataset(dataset='awa2', num_classes=50):
    """
    Generates a dataset of hypernym relations for the class synsets
    It includes every relation between parent and child synsets for
    the whole hierarchy of the class synsets
    Retruns:
        hypernyms - all hypernyms relations as 
                    pairs of ids corresponsing to
                    ids in name2index
        name2index  - map of (synset_name, synset_id)
    """
    if dataset == 'awa2':
        class_synsets_f='data/awa2/class_synsets.txt'
    else:
        class_synsets_f='data/imagenet/class_synsets.txt'

    from nltk.corpus import wordnet as wn

    synset_names = np.loadtxt(class_synsets_f, dtype=str)

    synset_names = synset_names[:num_classes] # take only first num_classes

    class_synsets = [wn.synset(s) for s in synset_names]

    name2index = {}
    for i, s in enumerate(class_synsets):
        name2index[s.name()] = i

    hypernyms = []
    idx = 0
    while idx < len(class_synsets):
        synset = class_synsets[idx]
        synset_it = synset
        while len(synset_it.hypernyms()) > 0:
            for h in synset_it.hypernyms():
                if h.name() not in name2index:
                    name2index[h.name()] = len(class_synsets)
                    class_synsets.append(h)

                hypernyms.append(
                    [name2index[synset.name()], name2index[h.name()]])
            synset_it = synset_it.hypernyms()[0]

        idx += 1
    
    hypernyms = np.array(hypernyms)

    return hypernyms, name2index

def generate_negative_hypernyms(N, max_int):
    """
    Generates N pairs of random ids 
    between [0, max_int)
    """
    return np.random.randint(0, max_int, (N,2))

def get_hypernyms_per_class(hypernyms, N_classes, N_words):
    """
        Transforms hypernyms dataset from list of pairs to
        a list of 'N_classes' one-hot encoded lists of size N_words
        having a 1 at the position of the id of all hypernyms 
        of that class. Eg. if class with id 4 has hypernyms 12 and 15 
        the list at position 4 will have 1s at position 4, 12 and 15 

        Returns:
        hypernums_per_class - transformed dataset
    """

    hypernyms_per_class = np.eye(N_classes, N_words)

    for [c, p] in hypernyms:
        is_base_class = c < N_classes
        
        if is_base_class:
            hypernyms_per_class[c, p] = 1
    
    return hypernyms_per_class


def save_hypernym_dataset(dataset, num_classes):
    """
    Saves the hypernyms dateset and name2index on disk so that 
    it can be loaded without needing nltk/wordnet
    """
    hypernyms, name2index = generate_hypernym_dataset(dataset, num_classes)

    if not os.path.exists('saved'):
        os.mkdir('saved')

    hyp_f = 'saved/%s-%d_hypernyms.pkl' % (dataset, num_classes)
    n2i_f = 'saved/%s-%d_name2index.pkl' % (dataset, num_classes)

    with open(hyp_f, 'wb') as out_f:
        pickle.dump(hypernyms, out_f)
    with open(n2i_f, 'wb') as out_f:
        pickle.dump(name2index, out_f)

def load_hypernym_dataset(dataset='awa2', num_classes=50):
    """
    Loads precomputed hypernym dataset
    """
    hyp_f = 'saved/%s-%d_hypernyms.pkl' % (dataset, num_classes)
    n2i_f = 'saved/%s-%d_name2index.pkl' % (dataset, num_classes)

    if (not os.path.exists(hyp_f) or 
            not os.path.exists(n2i_f)):
        save_hypernym_dataset(dataset, num_classes)

    with open(hyp_f, 'rb') as in_f:
        hypernyms = pickle.load(in_f)
    with open(n2i_f, 'rb') as in_f:
        name2index = pickle.load(in_f)

    return hypernyms, name2index
    

