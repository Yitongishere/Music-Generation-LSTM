import numpy as np
import json
import music21 as m21
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH

# 设置music21环境，使得乐谱可以通过第三方软件musescore打开
env = m21.environment.Environment()
m21.environment.Environment()['musicxmlPath'] = r'D:\MuseScore\bin\MuseScore3.exe'
m21.environment.Environment()['musescoreDirectPNGPath'] = r'D:\MuseScore\bin\MuseScore3.exe'


class MelodyGenerator:
    """
    基于搭建的LSTM模型，提供旋律生成功能。并可实现编码符号音乐 -> midi音乐格式的转换
    """
    def __init__(self, model_path="./weights/best_model.pth.tar", mapping_path="mapping.json"):
        # 初始化tensorflow模型

        # 加载预处理过程中生成的mapping文件（编码-int 对照表）
        with open(mapping_path, "r") as fp:
            self.mapping = json.load(fp)

        # 加载训练好的模型
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        # 用“/”补齐未满足序列长度的旋律“动机”
        self.start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """
        通过提供给定的旋律“动机”，即旋律开头，通过训练的网络模型，进行旋律续写。
        可设定续写的旋律长度，输入模型的“动机”最大长度，以及生成过程的不确定性（serendipity）。

        :param seed: str，旋律“动机”，根据此动机进行续写。需要符合编码规则，空格为分隔符
        :param num_steps: int，指定旋律续写的step数目，每个step代表16分音符
        :param max_sequence_length: 向模型输入的旋律“动机”的最大长度
        :param temperature: float，(0,1].不确定性参数，越大则生成的旋律不确定性越强
        :return melody：list，生成的旋律，以编码字符形式表示
        """
        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self.start_symbols + seed

        # map seed to int
        seed = [self.mapping[token] for token in seed]

        for i in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # onehot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self.mapping))
            # (1, max_sequence_length, vocabulary_size)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self.sample_with_temperature(probabilities, temperature)

            # update the seed
            seed.append(output_int)

            # map int to encoding
            output_token = [k for k, v in self.mapping.items() if v == output_int][0]

            # check whether we're at the end of melody
            if output_token == "/":
                break

            # update the melody
            melody.append(output_token)

        return melody



    def sample_with_temperature(self, probabilities, temperature):
        """
        根据temperature对预测结果进行采样。
        当temperature -> 0, 采样概率中最大的一项即为1，其余为0
        当temperature = 1, 遵循模型预测的采样概率
        当temperature -> 无穷大, 采样概率为均匀分布
        :param probabilities: 概率向量，代表模型预测的下一个音为对应音符的概率
        :param temperature: 使用该参数调整模型预测概率与实际采样概率的关系
        :return index: int，最终被预测的下一个音的结果（独热向量索引）
        """
        predictions = np.log(probabilities + 1e-6) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index


    def save_melody(self, melody, format="midi", file_name="melody.mid", step_duration=0.25):

        """
        将生成的旋律从编码格式转换为midi格式，生成可播放的.mid文件。

        :param melody: list of str，字符编码形式的旋律
        :param format:  生成的文件格式
        :param file_name:   生成的文件名
        :param step_duration:   以4分音符为单位的时值下，编码中每一个step的长度
        :return:
        """

        # 创建 music21 stream, 使用默认4/4，Cmaj
        stream = m21.stream.Stream()

        # 根据生产的旋律，建立音符、休止符object
        start_token = None
        step_counter = 1
        for i, token in enumerate(melody):
            # 对于一个音符或者休止符开始的第一个step，即非"_"情况
            # 代表上一事件结束，整合入 music21 stream
            if token != "_" or i == len(melody) - 1:       # 需考虑旋律结尾处最后一个音的整合
                if start_token is not None:
                    quarter_length_duration = step_duration * step_counter
                    # 休止符
                    if start_token == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # 音符
                    else:
                        m21_event = m21.note.Note(int(start_token), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # 重置时值计数器
                    step_counter = 1
                # 对上一事件的整合进入music21 stream工作结束，开始处理当前位置的音符或休止符
                start_token = token

            # 对于一个音符或者休止符开始的延长符号，即"_"情况
            else:
                step_counter += 1

        # 创建midi文件
        stream.write(format, file_name)



if __name__ == "__main__":
    mapping_chn = "mapping_chn.json"
    mapping_gen = "mapping_gen.json"
    # chinesefolksong_generator = MelodyGenerator(model_path="./weights/chn_folksong_accuracy 0.8932.h5", mapping_path=mapping_chn)
    germanfolksong_generator = MelodyGenerator(model_path="./weights/gen_folksong_accuracy 0.9090.h5", mapping_path=mapping_gen)
    seed1 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    seed2 = "60 _ _ _ _ _ _ _ _ _ 69 _ _ 67 67 69"  # 切分节奏
    seed3 = "59 _ _ _ _ _ 60 60 60 _ _ _ _ _ 62 62" # 附点节奏
    # chinesefolksong_melody = chinesefolksong_generator.generate_melody(seed1, 500, SEQUENCE_LENGTH, 0.8)
    germanfolksong_melody = germanfolksong_generator.generate_melody(seed1, 500, SEQUENCE_LENGTH, 0.8)
    # print("Encoded Chinese folk song melody: \n", chinesefolksong_melody)
    print("Encoded German folk song melody: \n", germanfolksong_melody)
    # chinesefolksong_generator.save_melody(chinesefolksong_melody, file_name="melodies/chn_folk.mid")
    germanfolksong_generator.save_melody(germanfolksong_melody, file_name="melodies/gen_folk.mid")


