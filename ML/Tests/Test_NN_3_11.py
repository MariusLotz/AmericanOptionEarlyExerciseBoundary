import ML.Base as B
import numpy as np
import tensorflow as tf
from tensorflow import keras

def test():
    file_path = '/home/user/PycharmProjects/AmericanOptionPricer/ML/Small_Sample_3_11'
    model = keras.models.load_model('/home/user/PycharmProjects/AmericanOptionPricer/ML/NN_of_Small_Sample_3_11')
    x_train, y_train = B.load_and_return_trainingsdata(file_path)
    B.test_model_on_training_data(model, x_train, y_train)


if __name__=="__main__":
    test()