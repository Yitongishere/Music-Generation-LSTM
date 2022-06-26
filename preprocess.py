import os
import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from utils import load

# 设置music21环境，使得乐谱可以通过第三方软件musescore打开
env = m21.environment.Environment()
m21.environment.Environment()['musicxmlPath'] = r'D:\MuseScore\bin\MuseScore3.exe'
m21.environment.Environment()['musescoreDirectPNGPath'] = r'D:\MuseScore\bin\MuseScore3.exe'

# 数据集路径
DATA_PATH = "deutschl_folksong/erk"
# 以16分音符为最小时间单位，筛去特殊长度音符。以4分音符为1拍表示。
ACCEPTABLE_NOTE_DURATION = [0.25,   # 16分音符
                            0.5,    # 8分音符
                            0.75,   #   8分音符带附点
                            1,      # 4分音符
                            1.5,    #   4分音符带附点
                            2,      # 2分音符
                            3,      #   2分音符带附点
                            4]      # 全音符
# 编码后的乐曲数据集路径
ENCODED_SAVE_PATH = "encoded_dataset_gen"
# 将编码后的乐曲存入同一文件
SINGLE_FILE_DATASET = "single_file_dataset_gen.txt"
# mapping文件的路径
MAPPING_PATH = "mapping.json"
# 输入模型的序列长度
SEQUENCE_LENGTH = 64


def load_songs(dataset_path):

    """
    加载文件夹里的所有音乐文件，即.krn文件

    :param dataset_path: 数据集路径，可包含子目录
    :return songs: 一个包含数据集中每首音乐的list，每个元素为一个object代表一首乐曲，由music21对.krn文件解析而成
    """

    # 遍历整个文件夹中的.krn文件，加载
    songs = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            # 筛选出包含音乐信息的.krn文件
            if file[-3:] == "krn":
                # 用music21解析.krn文件
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs


def check_note_duration(song, acceptable_note_duration):

    """
    筛查是否该首乐曲中所包含的音符和休止符都属于我们设定的“可接受的时值”的范围内

    :param song: 一首被music21所解析的的音符和休止符都有可接受的时值song object
    :param acceptable_note_duration: 一个list，在该模型中，设定的可被接受的时值的集合
    :return: boolean: 是否该首乐曲中所包含的音符和休止符都有可接受的时值
    """

    # 只检查被music21所解析的乐曲object中还包含拍号、调号等信息，我们只选择其中的音符（notes）和休止符（rests）
    for note in song.flat.notesAndRests:
        # 以4分音符为时值长度单位“1”
        if note.duration.quarterLength not in acceptable_note_duration:
            return False
    return True


def transpose_to_Cmaj_amin(song):

    """
    将数据集中不同调的乐曲都转为同一个调号。即将所有的大调转为C大调（Cmaj），
    将所有的小调转为a小调（amin）。原因是全部的24个调中，大部分都非常罕见，
    罕见的调中包含的一些变化音也很少会出现，将其转为Cmaj和amin可以更有效地运用数据集，

    :param song: 一首被music21所解析的的音符和休止符都有可接受的时值song object
    :return trasposed_song：转调后的song
    """

    # 使用music21获取乐曲的调号:
        # song object的关系是Score（song）中包含Parts包含measures，measures中包含调号
        # 调号为第一个part的第一个measure中第四个元素
    part0 = song.getElementsByClass(m21.stream.Part)[0]
    measures0_part0 = part0.getElementsByClass(m21.stream.Measure)[0]
    key = measures0_part0[4]

    # 有时候乐曲中未包含调号信息，可用music21进行estimate
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # 获得调号信息后，判断出该乐曲转调为Cmaj或amin需要移动多少个半音（interval for transpose）
        # 先用mode判断大调或小调
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # 根据interval进行转调操作
    trasposed_song = song.transpose(interval)

    return trasposed_song


def encoding_m21_to_string(song, time_step=0.25):

    """
    将被music21解析的song object中的音符和休止符转化为时间序列表示，音符音高使用midi音高，
    用"_"表示保持，组成”音高+时值“的表示方法。休止符用“r”表示。分隔符为空格。
    e.g. 四分音符的midi音高为60的do表示为："60 _ _ _"
        八分音符休止符表示为： "r _"
        四分音符的midi音高为64的mi和八分音符的midi音高为65的fa表示为： "64 _ _ _ 65 _"
    将单首乐曲song编码后以字符串的形式返回。

    :param song: 一首被music21所解析的的音符和休止符都有可接受的时值song object
    :param time_step: 16分音符的表示，若以quarterLength表示，即为 (16th note Length / Quarter Length) = 0.25
    :return encoded_song: string，该首乐曲编码后的字符串
    """

    encoded_song = []

    for event in song.flat.notesAndRests:
        # 将音符用midi音高来表示
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # 将休止符用字符“r”来表示
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # 将音符和休止符转换成包含时值信息的time series representation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # 将编码后的“list”，转为string,空格为分隔符
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def create_single_file_dataset(encoded_save_path, single_file_dataset_path, sequence_length=SEQUENCE_LENGTH):

    """
    将数个乐曲编码文件中的内容整合为一个”长“字符串，存入单个文件中，作为单个文件数据集。其中每首乐曲直接用分隔符隔开，
    分隔符为sequence_length个”/“，表示一首乐曲的结束。

    :param encoded_save_path: 存放每首歌曲编码文件的目录的路径
    :param single_file_dataset_path: 将数据集内所有乐曲编码后的并整合起来存入的文件的路径
    :param sequence_length: 输入模型的序列长度
    :return songs: 将数据集内所有乐曲编码后的并整合起来，且包含隔符隔的字符串
    """

    # 在每首曲子结束插入分隔符，分隔符个数 = 输入模型的序列长度
    delimiter = "/ " * sequence_length

    # 创建一个空字符串，将每首歌的字符串编码、分隔符整合进来
    songs = ""
    for path, _, files in os.walk(encoded_save_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + delimiter
    # 删去最后的空格
    songs = songs[:-1]

    # 把汇集了所有乐曲的字符串存入单个文件
    with open(single_file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path=MAPPING_PATH):

    """
    在将编码后的乐曲信息送入模型前，需将字符串信息转换为int，这时需要创建一个mapping文件，类似于”查询表“，
    查找每个编码后的字符对应什么int，以此创建词汇表(vocabulary)，以完成token向int的转换。

    :param songs: 整合了所有乐曲编码的字符串
    :param mapping_path: 生成的mapping.json文件的路径
    :return:
    """

    mappings = {}
    # 在数据集中找到每一种token，创建vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, token in enumerate(vocabulary):
        mappings[token] = i

    # 将vocabulary存入json文件
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

    return


def mapping_convert(songs):

    """
    以mapping.json文件作为“查询表”，将以字符串形式储存的音符、休止符信息转换为以整数形式储存的list

    :param songs: 整合了所有乐曲编码的字符串
    :return songs_mapped: 通过mapping.json转换为整数数组的音乐编码信息
    """

    # 加载mapping文件
    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)
    # 建立空list，存放str被mapped之后的数字信息
    songs_mapped = []
    # 将str转换为list
    songs = songs.split()
    # 取songs中的符号一一在mapping中对应转换
    for token in songs:
        mapped_token = mapping[token]
        songs_mapped.append(mapped_token)

    return songs_mapped


def generate_training_data(encoded_dataset="single_file_dataset.txt", sequence_length=SEQUENCE_LENGTH):

    """
    根据被转换为int的音乐信息，转化为独热向量后，生成匹配模型的训练数据。
    inputs -> (Number of sequence, sequence length, vocabulary_size)
    targets -> (Number of sequence)

    :param songs_mapped: 通过mapping.json转换为整数数组的音乐编码信息
    :return: inputs: list, 输入数据。 outputs: list, 真实标签
    """

    inputs = []
    targets = []

    # 加载mapping文件
    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)
    # 词表长度
    vocabulary_size = len(mapping)

    # 加载编码后的整个数据集
    songs = load(encoded_dataset)
    # 将str转换为int list
    songs_mapped = mapping_convert(songs)

    # 根据输入序列长度计算输入序列数量
    num_sequence = len(songs_mapped) - sequence_length
    # 生成输入序列，真实标签
    for i in range(num_sequence):
        input_i = songs_mapped[i : i + sequence_length]
        inputs.append(input_i)
        target_i = songs_mapped[i + sequence_length]
        targets.append(target_i)
    # 转为独热向量表示： (序列数量，序列长度) -> (序列数量，序列长度，词表长度)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    # inputs和targets数据类型统一
    targets = np.array(targets)

    return inputs, targets


def preprocess(dataset_path, acceptable_note_duration=ACCEPTABLE_NOTE_DURATION):

    """
    打包数据预处理的整个一套流程：
        加载 -> 筛选 -> 转调 -> 编码 -> 整合为单个文件 -> mapping(token to int) -> 生成‘input-target’数据（独热向量）
    :param dataset_path:
    :return inputs, output (numpy.array): 用于模型训练的“输入-标签”数据
            inputs: 3d (数据条数，序列长度，词表长度)
            targets： 1d (数据条数)
    """

    print("PREPROCESS IN PROGRESS >>>\n-----------------------------------------")
    # 加载数据集
    print("Loading music ...")
    songs = load_songs(dataset_path)
    print(f"{len(songs)} songs loaded.")

    # 以每首乐曲为单位处理
    for i, song in enumerate(songs):
        # 以16分音符为最小的时间单位，筛去包含特殊音长的乐曲
        if not check_note_duration(song, acceptable_note_duration):
            continue

        # 将所有音乐转调，大调转为C大调，小调转为a小调。即不包含仍和升降号
        song = transpose_to_Cmaj_amin(song)

        # 将读取的音乐编码为music time series representation, 将object转换为string
        encoded_song = encoding_m21_to_string(song)

        # 每篇乐曲分别存入单个文件中
        save_path = os.path.join(ENCODED_SAVE_PATH, str(i))
        with open(save_path, 'w') as f:
            f.write(encoded_song)

    # 将编码后的多个txt文件整合存入单个txt文件中
    songs = create_single_file_dataset(ENCODED_SAVE_PATH, SINGLE_FILE_DATASET)

    # 建立mapping查询表
    create_mapping(songs)

    print("PREPROCESS FINISHED >>>\n-----------------------------------------")



if __name__ == "__main__":
    preprocess(DATA_PATH)


