import csv
import sys
import numpy as np

emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?: '
ALPHABET_SIZE = len(emb_alphabet)
# associate each character in our alphabet to a number:
CHAR_DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}

# This is used to sample the data add feed batches to the keras model
# this is only necessary when the data is represented as one hot vectors
# in the alphabet space. This is because the entire data set cannot
# be stored in RAM with such a sparse representation

class BatchLoader(object):


    def __init__(self, path, truncate_len, train_val_split=0.9, are_labels="True"):

        self.path = path
        self.cur_spot = 0
        self.are_labels = are_labels
        self.truncate_len = truncate_len
        self.num_examples = 0
        self.split = train_val_split

        self.train_data = {}
        self.val_data = {}

        data = self.read_csv_data()
        div = int(self.num_examples*self.split)

        if are_labels:
            self.train_data['X'] = data['X'][:div]
            self.train_data['Y'] = data['Y'][:div]

            self.val_data['X'] = data['X'][div:]
            self.val_data['Y'] = data['Y'][div:]

        self.test_data = data


    # read from the csv file and returns a dictionary of the X and Y values
    def read_csv_data(self):

        csv.field_size_limit(sys.maxsize)

        with open(self.path) as csvfile:
            reader = csv.DictReader(csvfile)

            comments = []
            labels = []
            self.num_examples = 0

            for row in reader:
                self.num_examples += 1
                comment = "-"
                comment += row['comment_text'].replace("\n"," ")
                comments.append(comment[:self.truncate_len])


                if self.are_labels:
                    label = [int(row['toxic']), int(row['severe_toxic']), int(row['obscene']), int(row['threat']),
                             int(row['insult']), int(row['identity_hate'])]
                    labels.append(label)


            if self.are_labels:
                dict =  {"X": np.asarray(comments), "Y":  np.asarray(labels)}
            else:
                dict = {"X": np.asarray(comments)}
            return dict


    # embed the comments as a one vector in the alphabet space
    def embed_comment_onehot(self, comments):

        truncate_len = self.truncate_len
        embedding = np.zeros((len(comments), truncate_len, ALPHABET_SIZE, 1))

        for i, comment in enumerate(comments):

            comment = comment.lower() # makes everything lower case, might not want to do

            if len(comment) > truncate_len:
                comment = comment[:truncate_len]

            for j,char in enumerate(comment):

                try:
                    CHAR_DICT[char]
                    embedding[i, j, CHAR_DICT[char], 0] = 1
                except Exception:
                    pass

        return embedding

    # embed the comments as a tokenized numbers where each numbder corresponds to a
    # letter of the alphabet
    def embed_comment_alphdict(self, comments):

        truncate_len = self.truncate_len
        embedding = np.zeros((len(comments), truncate_len, 1))

        for i, comment in enumerate(comments):

            comment = comment.lower() # makes everything lower case, might not want to do

            if len(comment) > truncate_len:
                comment = comment[:truncate_len]

            for j,char in enumerate(comment):

                try:
                    CHAR_DICT[char]
                    embedding[i, j, 0] = CHAR_DICT[char]
                except Exception:
                    pass

        return embedding

    # takes the data in dict form and shuffels
    def shuffle_train_data(self):
        random_mask = np.random.permutation(self.train_data['X'].shape[0])
        self.train_data['X'] = self.train_data['X'][random_mask]
        self.train_data['Y'] = self.train_data['Y'][random_mask]


    # iterator method for train set
    def iterate_minibatch_train(self, batch_size):
        i = 0
        self.shuffle_train_data()

        while 1:

            if (i + batch_size > self.train_data['X'].shape[0]):
                x_batch = self.train_data['X'][i:]
                x_batch = self.embed_comment_onehot(x_batch)
                y_batch = self.train_data['Y'][i:]
                i = 0
                self.shuffle_train_data()

            else:
                x_batch = self.train_data['X'][i:i+batch_size]
                x_batch = self.embed_comment_onehot(x_batch)
                y_batch = self.train_data['Y'][i:i+batch_size]
                i += batch_size

            yield x_batch, y_batch

    # iterator method for validation set
    def iterate_minibatch_val(self, batch_size):
        i = 0

        while 1:

            if (i + batch_size > self.val_data['X'].shape[0]):
                x_batch = self.val_data['X'][i:]
                x_batch = self.embed_comment_onehot(x_batch)
                y_batch = self.val_data['Y'][i:]
                i = 0

            else:
                x_batch = self.val_data['X'][i:i+batch_size]
                x_batch = self.embed_comment_onehot(x_batch)
                y_batch = self.val_data['Y'][i:i+batch_size]
                i += batch_size
            yield x_batch, y_batch

    # iterator method for test set
    def iterate_minibatch_test(self, batch_size):
        i = 0
        print("called test")
        while 1:
            print("called2 test ", i)
            if (i + batch_size > self.test_data['X'].shape[0]):
                x_batch = self.test_data['X'][i:]
                x_batch = self.embed_comment_onehot(x_batch)
                i = 0
                print("FINISHED ONE TEST EPOCH.")
            else:
                x_batch = self.test_data['X'][i:i+batch_size]
                x_batch = self.embed_comment_onehot(x_batch)
                i += batch_size

            yield x_batch
