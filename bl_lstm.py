# BASELINE LSTM
import pandas as pd

from keras.models import Model
from keras.layers import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000 # max features before embedding
maxlen = 900 # max length of comment, greater than is truncated
embed_size = 128 # size of embedding space

# read data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train = train.sample(frac=1)

# csv to python
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

# clean up data, truncate according to maxlen and remove chars not in alphabet
emb_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWYXZ0123456789-,;.!?:" '
def clean_up(data):
    print("cleaning data...")
    for i in range(len(data)):
        data[i] = data[i].replace("\n", " ")
        data[i] = data[i][:maxlen]
        for j,char in enumerate(data[i]):

            if char not in emb_alphabet:
                data[i] = data[i].replace(char,"")

    return data

list_sentences_train = clean_up(list_sentences_train)
list_sentences_test = clean_up(list_sentences_test)

# tokenize data
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

# this is the baseline LSTM model

# create model
def LSTM_Model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    #x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x) # if training on gpu, sub for line above
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = LSTM_Model()
batch_size = 128
epochs = 4
file_path="weights_base.best.hdf5"


print("FITTING...")
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

print("TESTING...")
model.load_weights(file_path)
y_test = model.predict(X_te, verbose=1, batch_size=batch_size)
sample_submission = pd.read_csv("./data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)
