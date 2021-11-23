import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import yaml
import time

class cifar10_cnn:
    def __init__(self):
        print("Initialization")
        param_file = open("../params/config.yaml", 'r')
        network_params = yaml.safe_load(param_file)
        # self.classes = network_params['classes']
        self.input_shape_x = network_params['input_shape_x']
        self.input_shape_y = network_params['input_shape_y']
        self.num_channels = network_params['num_channels']
        self.input_shape = (28, 28, 1)
        self.batch_size = network_params['batch_size']
        self.num_classes = network_params['num_classes']
        self.num_epochs = network_params['num_epochs']
        self.learning_rate = network_params['learning_rate']
        self.decay = network_params['decay']
        self.dropout_value_layer_1 = network_params['dropout_value_layer_1']
        self.dropout_value_layer_2 = network_params['dropout_value_layer_2']
        self.dropout_value_layer_3 = network_params['dropout_value_layer_3']
        self.num_nurons_layer_2 = network_params['num_nurons_layer_2']
        self.num_nurons_layer_3 = network_params['num_nurons_layer_3']
        self.act_func_layer_2 = network_params['act_func_layer_2']
        self.act_func_layer_3 = network_params['act_func_layer_3']
        self.act_func_softmax_layer = network_params['act_func_softmax_layer']
        self.loss_function = network_params['loss_function']
        self.validation_split = network_params['validation_split']
        self.regularization_method = network_params['regularization_method']
    
    def get_dataset(self):
        print("Preparing dataset!")
        dataset_raw = tf.keras.datasets.mnist

        (self.training_data, self.training_label), (self.test_data, self.test_label) = dataset_raw.load_data()
        self.training_label = self.training_label.flatten()
        self.test_label = self.test_label.flatten()

        self.training_data = self.training_data.reshape(self.training_data.shape[0], self.training_data.shape[1], self.training_data.shape[2], 1)
        self.training_data = self.training_data / 255.0
        self.test_data = self.test_data.reshape(self.test_data.shape[0], self.test_data.shape[1], self.test_data.shape[2], 1)
        self.test_data = self.test_data / 255.0

        self.training_label = tf.one_hot(self.training_label.astype(np.int32), depth=10)
        self.test_label = tf.one_hot(self.test_label.astype(np.int32), depth=10)

        return self.training_data, self.test_data, self.training_label, self.test_label
    
    def build_cnn_model(self):
        print("Building the model!")
        if self.regularization_method == 'dropout':
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(self.input_shape_x, (5,5), padding='same', activation=self.act_func_layer_2, input_shape=self.input_shape),

                tf.keras.layers.Conv2D(self.input_shape_y, (5,5), padding='same', activation=self.act_func_layer_2),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Dropout(self.dropout_value_layer_1),

                tf.keras.layers.Conv2D(self.num_nurons_layer_2, (3,3), padding='same', activation=self.act_func_layer_3),

                tf.keras.layers.Conv2D(self.num_nurons_layer_2, (3,3), padding='same', activation=self.act_func_layer_3),
                tf.keras.layers.MaxPool2D(strides=(2,2)),
                tf.keras.layers.Dropout(self.dropout_value_layer_2),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.num_nurons_layer_3, activation=self.act_func_layer_3),
                tf.keras.layers.Dropout(self.dropout_value_layer_3),
                tf.keras.layers.Dense(self.num_classes, activation=self.act_func_softmax_layer)
        ])

        if self.regularization_method == 'l2':
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(self.input_shape_x, (5,5), padding='same', activation=self.act_func_layer_2, input_shape=self.input_shape, kernel_regularizer='l2'),

                tf.keras.layers.Conv2D(self.input_shape_y, (5,5), padding='same', activation=self.act_func_layer_2, kernel_regularizer='l2'),
                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Conv2D(self.num_nurons_layer_2, (3,3), padding='same', activation=self.act_func_layer_3, kernel_regularizer='l2'),

                tf.keras.layers.Conv2D(self.num_nurons_layer_2, (3,3), padding='same', activation=self.act_func_layer_3, kernel_regularizer='l2'),
                tf.keras.layers.MaxPool2D(strides=(2,2)),
                

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.num_nurons_layer_3, activation=self.act_func_layer_3, kernel_regularizer='l2'),
                tf.keras.layers.Dense(self.num_classes, activation=self.act_func_softmax_layer, kernel_regularizer='l2')
        ])
        return self.model
    
    def train_cnn_model(self):
        print("Training!")
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, decay=self.decay), loss=self.loss_function, metrics=['acc'])
        self.H = self.model.fit(self.training_data, self.training_label, batch_size=self.batch_size, epochs=self.num_epochs, validation_split=self.validation_split)
        return self.H
    
    def evaluate_cnn_model(self):
        print("Evaluating the model!")
        self.validation_loss, self.validation_accuracy = self.model.evaluate(self.test_data, self.test_label)
        print('Validation Loss: ' + str(self.validation_loss))
        print('Validation Accuracy: ' + str(self.validation_accuracy))

        return self.validation_loss, self.validation_accuracy
    
    def predict(self):
        print("Predicion")
        self.predicted_label = self.model.predict(self.test_data)
        return self.predicted_label

    def plot_results(self):
        epoch_vector = np.arange(0, self.num_epochs)

        plt.figure()
        # Plotting Training LOSS
        plt.plot(epoch_vector, self.H.history['loss'], label='training_loss_value')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid()
        # plt.show()

        # Plotting Validation Loss
        plt.figure()
        plt.plot(epoch_vector, self.H.history['val_loss'], label='validation_loss_value')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid()
        # plt.show()

        # Plotting Training Accuracy
        plt.figure()
        plt.plot(epoch_vector, self.H.history['acc'], label='training_accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Trining Accuracy')
        plt.legend()
        plt.grid()
        # plt.show()

        # Plotting Validation Accuracy
        plt.figure()
        plt.plot(epoch_vector, self.H.history['val_acc'], label='validation_accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

        # compute the confusion matrix
        confusion_matrix = tf.math.confusion_matrix(np.argmax(self.test_label,axis = 1), np.argmax(self.predicted_label,axis = 1))
        print(confusion_matrix)
        plt.figure(figsize=(10, 8))
        plt.title('Confusion Matrix')
        heat_map = sns.heatmap(confusion_matrix, annot=True, fmt='g')
        plt.show()