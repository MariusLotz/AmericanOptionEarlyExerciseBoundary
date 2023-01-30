import tensorflow as tf
import os

def loading_from_file_boundary():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('training_data', 'r')
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

        fx_list.append(xline[3]) # exercise boundary (11,1)
    file.close()
    return x_list, fx_list

def loading_from_file_price():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    file = open('training_data', 'r')
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
             xline[2],  # sigma
             xline[4][0]] # s
        x_list.append(x)

        fx_list.append([xline[4][1]]) # price (1,1)
    file.close()
    return x_list, fx_list

def training_model_for_boundary():
    x_list, fx_list = loading_from_file_boundary()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_shape=(3,), activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(11, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics='mean_absolute_percentage_error')
    nepoch = 1000
    nbatch = 64
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save("test_model_boundary")

def training_model_for_price():
    x_list, fx_list = loading_from_file_price()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_shape=(4,), activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics='mean_absolute_percentage_error')
    nepoch = 1000
    nbatch = 64
    model.fit(x_list, fx_list, epochs=nepoch, batch_size=nbatch)
    model.save("test_model_price")

if __name__ == "__main__":
    #training_model_for_boundary()
    training_model_for_price()