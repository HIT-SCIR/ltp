import time
import functools

from ltp_extension.algorithms import StnSplit


def clock(func):
    """this is outer clock function"""

    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function"""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        print(func.__name__ + "({}s)".format(time_cost))
        return result

    return clocked  # --> 3


def convert2raw(input_file, output_file):
    with open(input_file) as fi:
        with open(output_file, "w") as fo:
            for line in fi:
                line = line.strip()
                line = line.split()
                line = "".join(line)
                fo.write(line + "\n")


@clock
def jieba_load():
    import jieba

    _force_load = list(jieba.cut("我来到北京清华大学"))
    return jieba


@clock
def ltp_cws_load(model_path):
    from ltp_extension.perceptron import CWSModel

    model = CWSModel.load(model_path)
    return model


@clock
def ltp_load(
    cws_model_path="../../data/models/cws_model.bin",
    pos_model_path="../../data/models/pos_model.bin",
    ner_model_path="../../data/models/ner_model.bin",
):
    from ltp_extension.perceptron import CWSModel, POSModel, NERModel

    cws_model = CWSModel.load(cws_model_path)
    pos_model = POSModel.load(pos_model_path)
    ner_model = NERModel.load(ner_model_path)
    return cws_model, pos_model, ner_model


@clock
def jieba_cut(model, sentences):
    return [list(model.cut(sentence)) for sentence in sentences]


@clock
def ltp_cut(model, sentences, threads=1):
    return model.batch_predict(sentences, threads)


def cws_benchmark():
    with open("../data/models/pku-all.txt") as fi:
        sentences = [line.strip() for line in fi.readlines()]

    # 1.1205673217773438e-05s
    jieba_model = jieba_load()
    # 3.067751884460449s
    ltp_model = ltp_cws_load("../data/models/cws_model.bin")

    # 35.27866315841675s
    jieba_cut(jieba_model, sentences)

    # 58.64469790458679s
    ltp_cut(ltp_model, sentences, threads=1)
    # 30.638360023498535s
    ltp_cut(ltp_model, sentences, threads=2)
    # 17.13770890235901s
    ltp_cut(ltp_model, sentences, threads=4)
    # 11.386134147644043s
    ltp_cut(ltp_model, sentences, threads=8)
    # 9.87795615196228s
    ltp_cut(ltp_model, sentences, threads=16)


@clock
def ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=1):
    batch_words = cws_model.batch_predict(sentences, threads)
    batch_pos = pos_model.batch_predict(batch_words, threads)
    batch_ner = ner_model.batch_predict(batch_words, batch_pos, threads)

    return batch_words, batch_pos, batch_ner


def pipeline_benchmark():
    with open("../data/models/pku-all.txt") as fi:
        sentences = [line.strip() for line in fi.readlines()]

    # 11.27102518081665s
    cws_model, pos_model, ner_model = ltp_load()

    # multi threads
    # 139.701030254364s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=1)
    # 75.1799750328064s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=2)
    # 42.96349382400513s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=4)
    # 29.339277029037476s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=8)
    # 26.281506061553955s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=16)


@clock
def simple_pipeline(cws_model, pos_model, ner_model, sentence):
    words = cws_model.predict(sentence)
    pos = pos_model.predict(words)
    ner = ner_model.predict(words, pos)

    return words, pos, ner


def simple_example():
    # 11.27102518081665s
    cws_model, pos_model, ner_model = ltp_load()

    # simple and single thread
    sentence = "他叫汤姆去拿外衣。"
    # 0.0003898143768310547s
    words, pos, ner = simple_pipeline(cws_model, pos_model, ner_model, sentence)

    print(words)
    print(pos)
    print(ner)


@clock
def ltp_split(model: StnSplit, sentences, threads=1):
    return model.batch_split(sentences, threads)


def stn_split():
    split = StnSplit()
    with open("../data/corpus/data.txt") as fi:
        sentences = []
        for line in fi.readlines():
            line = line.strip()
            if line:
                sentences.append(line)

    # ltp_split(0.024084091186523438s)
    ltp_split(split, sentences, threads=1)
    # ltp_split(0.015919923782348633s)
    ltp_split(split, sentences, threads=2)
    # ltp_split(0.0134429931640625s)
    ltp_split(split, sentences, threads=4)
    # ltp_split(0.012565135955810547s)
    ltp_split(split, sentences, threads=8)
    # ltp_split(0.012076854705810547s)
    ltp_split(split, sentences, threads=16)


def main():
    simple_example()
    # stn_split()
    # cws_benchmark()
    # pipeline_benchmark()


if __name__ == "__main__":
    main()
