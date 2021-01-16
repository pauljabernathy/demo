from analyzer_base import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
DEFAULT_NUM_EPOCHS = 100


class DNNAnalyzer(AnalyzerBase):

    def print_gpu_blather(self):
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        #physical_devices = tf.config.list_physical_devices('GPU')
        #print(physical_devices)
        # tf.config.list_physical_devices('GPU') is giving an error message
        # The version of tensorflow I am using seems to no longer have that.

    def run_analysis(self, model, num_epochs=DEFAULT_NUM_EPOCHS, x_columns=None, y_column=None):
        if x_columns is not None and y_column is not None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_columns,
                y_column, test_size=.33, random_state=1)

        y_binary = to_categorical(self.y_train)
        model.fit(self.x_train, y_binary, epochs=num_epochs, verbose=2)
        dnn_predictions = model.predict(self.x_test)
        y_test_binary = to_categorical(self.y_test)
        pred = np.apply_along_axis(lambda r: 1 if r[1] > r[0] else 0, 1, dnn_predictions)
        confusion_matrix(self.y_test, pred)
        print(accuracy_score(self.y_test, pred))
        #print(classification_report(self.y_test, pred))
        print(pd.Series(pred).value_counts() / len(pred))

    def analyze_1(self, num_epochs=DEFAULT_NUM_EPOCHS):
        model = Sequential()
        model.add(Dense(9, input_dim=9, activation='relu'))
        model.add(Dense(9, input_dim=9, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.run_analysis(model, num_epochs)

    def analyze_2(self, num_epochs=DEFAULT_NUM_EPOCHS):
        model = Sequential()
        model.add(Dense(18, input_dim=9, activation='relu'))
        model.add(Dense(18, input_dim=9, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.run_analysis(model, num_epochs)

    def analyze_3(self, num_epochs=DEFAULT_NUM_EPOCHS):
        model = Sequential()
        model.add(Dense(27, input_dim=9, activation='relu'))
        model.add(Dense(27, input_dim=9, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.run_analysis(model, num_epochs)

    def analyze_4(self, num_epochs=DEFAULT_NUM_EPOCHS):
        model = Sequential()
        model.add(Dense(18, input_dim=9, activation='relu'))
        model.add(Dense(18, input_dim=18, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.run_analysis(model, num_epochs)

    def analyze_5(self, num_epochs=DEFAULT_NUM_EPOCHS, x_columns=None, y_column=None):
        model = Sequential()
        model.add(Dense(18, input_dim=9, activation='relu'))
        model.add(Dense(18, input_dim=18, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.run_analysis(model, num_epochs, x_columns, y_column)


if __name__ == '__main__':
    import tensorflow as tf
    tf.debugging.set_log_device_placement(True)
    dnn = DNNAnalyzer()
    dnn.analyze_1(57)
    dnn.print_gpu_blather()
    dnn.analyze_1(57)
    #dnn.analyze_2()
    #dnn.analyze_3()
    #dnn.analyze_4()



