# CHARACTER-LEVEL ATTENTION MODEL

import pandas as pd
from utils import *

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint

print("STARTING....")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 2000
max_features = 20000

def multi_conv_concat(x,widths,num_filters):

    out = None

    for i in range(len(widths)):


        cur_out = Conv2D(num_filters[i], (widths[i], ALPHABET_SIZE), padding="same")(x)
        cur_out = MaxPooling2D(pool_size=(2,1))(cur_out)
        #cur_out = keras.backend.squeeze(cur_out,2)

        if i == 0:
            out = cur_out
        else:
            out = keras.backend.concatenate([out, cur_out])

    return out



def get_model():

    inp = Input((maxlen,ALPHABET_SIZE,1))

    widths =      [2, 4, 8, 16,32,64,128]
    num_filters = [100,50,50,50,20,20,5]

    x = Lambda(multi_conv_concat, arguments={'widths':widths,'num_filters':num_filters})(inp)
    x = Reshape( (int(x.shape[1]),int(x.shape[2]*x.shape[3]) ) )(x)
    x = CuDNNLSTM(130, return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

print("INIT MODEL...")
model = get_model()
batch_size = 128
epochs = 12
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early]

print("READING DATA...")
BL_train = BatchLoader("./data/train.csv", maxlen, are_labels=True)
train_steps = BL_train.train_data['X'].shape[0] // batch_size
val_steps =   BL_train.val_data['X'].shape[0] // batch_size

print("FITTING MODEL...")
model.fit_generator(BL_train.iterate_minibatch_train(batch_size),
                    epochs=epochs,
                    validation_data=BL_train.iterate_minibatch_val(batch_size),
                    callbacks=callbacks_list,
                    steps_per_epoch=train_steps,
                    validation_steps=val_steps)

print("TESTING...")
model.load_weights(file_path)
BL_test = BatchLoader("./data/test.csv", maxlen, are_labels=False)
batch_size = 256
test_steps = int((BL_test.test_data['X'].shape[0])/ batch_size) + 1
y_test = model.predict_generator(BL_test.iterate_minibatch_test(batch_size), steps=test_steps, verbose=1)

print("SAVING PREDICTION...")
sample_submission = pd.read_csv("./data/sample_submission.csv")
print(len(y_test), len(sample_submission[list_classes]) )
sample_submission[list_classes] = y_test
sample_submission.to_csv("ans.csv", index=False)