import tensorflow as tf
import os
import numpy as np
import tensorflow.keras.losses as losses

def load_data():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('exercise_curve_train', 'r')
    x_list = []
    fx_list = []
    count = 0
    while True:
        line = file.readline()
        count += 1
        if not line:
            break
        try:
            xline = eval(line)
        except:
            continue
        x = [xline[0],  # r
             xline[1],  # q
             xline[2]]  # sigma
        x_list.append(x)
        fx_list.append(xline[3]) # exercise boundary (25,1)
    file.close()
    return x_list, fx_list

@tf.keras.utils.register_keras_serializable()
class PutCallSymmetryLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        n = 25
        def loss_func(true_pred):
            true, pred = true_pred[0], true_pred[1]
            error_put = pred[:n] - true[:n]
            error_call = pred[n:] - true[n:]
            return tf.reduce_mean(tf.square(error_put - error_call) + tf.abs(error_put) + tf.abs(error_call))
        losses = tf.map_fn(loss_func, (y_true, y_pred), dtype=tf.float32)
        return tf.reduce_mean(losses)

def main():

    # Register die PutCallSymmetryLoss-Funktion
    K.losses.put("PutCallSymmetryLoss", PutCallSymmetryLoss())
    x_list, fx_list = load_data()

    """## creating model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(3,), activation='relu'))
    tf.keras.layers.Dropout(0.2)
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    tf.keras.layers.Dropout(0.1)
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='linear'))
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    model_path = "model2_1"
    model.save(model_path)

    ## training model:
    ## 
    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 512
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 64
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 2048
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 50
    nbatch = 16
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 3000
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 128
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='log_cosh', metrics='mean_absolute_percentage_error')
    nepoch = 200
    nbatch = 1028
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)"""

    model_path = "model2_1"
    #model.compile(optimizer='adam', loss=putcallsymmetryloss)
    with K.utils.custom_object_scope({'PutCallSymmetryLoss': PutCallSymmetryLoss()}):
        model = tf.keras.models.load_model(model_path)
   
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='PutCallSymmetryLoss', metrics='mean_absolute_percentage_error')
    nepoch = 200
    nbatch = 128
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss=PutCallSymmetryLoss(), metrics='mean_absolute_percentage_error')
    nepoch = 200
    nbatch = 512
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model_path = "model2_1"
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss=PutCallSymmetryLoss(), metrics='mean_absolute_percentage_error')
    nepoch = 200
    nbatch = 1028
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    
if __name__=="__main__":
    main()