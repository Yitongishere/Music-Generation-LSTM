import matplotlib.pyplot as plt

def load(path):
    """
    读取文件通用函数
    :param path: 文件路径
    :return: 读取结果
    """
    with open(path, "r") as fp:
        song = fp.read()
    return song


def plot_history(history):
    """
    训练曲线绘制函数。
    :param history: history for model.fit (tensorflow)
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy plot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create loss plot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].set_ylabel("loss")
    axs[1].set_xlabel("epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")

    plt.savefig("learning_curve.png")
    plt.show()