import tensorflow.keras as keras

def build_the_model(output_units, hidden_size, loss, lr):
    """
    build and compile model

    :param output_units: LSTM输出向量维度
    :param hidden_size: LSTM隐藏层维度，list，可扩展至多层LSTM
    :param loss:        损失函数
    :param lr:          学习率
    :return:    model (tensorflow model)
    """

    # build model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(hidden_size[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LSTM(hidden_size[1])(x)
    # x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])

    model.summary()

    return model

