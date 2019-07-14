from sklearn import feature_extraction
from sklearn import feature_selection
from sklearn import decomposition
import jieba
from sklearn import preprocessing


def stand():
    """
    归一化处理
    :return: none
    """
    value = [[10, 30, 80], [0, 0.5, 1], [1, 1, 2]]
    std = feature_selection.VarianceThreshold(200)
    print(std.fit_transform(value))

def hanzivec():

    text_china = "如果有来生，要做一棵树，站成永恒。没有悲欢的姿势， 一半在尘土里安详，一半在风里飞扬；一半洒落荫凉，一半沐浴阳光。非常沉默、非常骄傲。从不依靠、从不寻找"
    text2 = "一棵树 骄傲"
    text_china_fen = ' '.join(list(jieba.cut(text_china)))
    dict1 = feature_extraction.text.CountVectorizer()
    dicttf = feature_extraction.text.TfidfVectorizer()
    data_han = dict1.fit_transform([text_china_fen])
    data_tf = dicttf.fit_transform([text_china_fen, text2])
    print(dict1.get_feature_names())
    print(data_han.toarray())
    print(dict1.get_feature_names())
    print(data_tf.toarray())


def dictvec():

    """
    数据的抽取
    :return: none
    """
    data1 = 'teacher is is pushing'
    data2 = 'student is paining'

    # 对字典实例化
    dict1 = feature_extraction.text.CountVectorizer()
    data_dict = dict1.fit_transform([data1, data2])

    print(dict1.get_feature_names())

    print(data_dict.toarray())


if __name__ == '__main__':
    stand()