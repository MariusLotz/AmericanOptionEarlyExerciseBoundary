import tensorflow as tf
import os

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

def main():
    x_list, fx_list = load_data()

    model_path = "model1_4"
    """## creating model 
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(3,), activation='relu'))
    tf.keras.layers.Dropout(0.2)
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Dense(50, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    
    model.save(model_path)

    ## training model:
    ## 
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 32
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 128
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    
    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 2048
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 50
    nbatch = 16
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 3000
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 100
    nbatch = 128
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 50
    nbatch = 2048
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 500
    nbatch = 512
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)"""

    model = tf.keras.models.load_model(model_path)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=10 ** (-6), patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='mean_absolute_percentage_error')
    nepoch = 999
    nbatch = 1048
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save(model_path)

if __name__=="__main__":
    main()