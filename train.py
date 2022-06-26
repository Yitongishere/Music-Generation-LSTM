import json
from preprocess import generate_training_data, SEQUENCE_LENGTH, SINGLE_FILE_DATASET, MAPPING_PATH
from model import build_the_model
from utils import plot_history

if __name__ == "__main__":

    # 生成训练数据
    inputs, targets = generate_training_data(encoded_dataset=SINGLE_FILE_DATASET, sequence_length=SEQUENCE_LENGTH)
    # 查看词汇表长度
    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)
    vocabulary_size = len(mapping)

    # 设定超参数
    loss = "sparse_categorical_crossentropy"
    output_units = vocabulary_size
    learning_rate = 0.001
    hidden_size = [256]
    epochs = 40
    batch_size = 64

    # 搭建模型
    model = build_the_model(output_units, hidden_size, loss, learning_rate)

    # 训练模型
    history = model.fit(x=inputs, y=targets, batch_size=batch_size, epochs=epochs)

    # 保存训练模型
    model.save("weights/model_weight.h5")

    # 画training curve
    plot_history(history)







