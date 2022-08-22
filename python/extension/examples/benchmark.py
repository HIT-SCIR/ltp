import functools
import time


def clock(func):
    """this is outer clock function."""

    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function."""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        print(func.__name__ + f"({time_cost}s)")
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


# ======================= 结巴分词 =======================


@clock
def jieba_load():
    import jieba

    jieba.initialize()
    return jieba


@clock
def jieba_cut(model, sentences):
    return [list(model.cut(sentence)) for sentence in sentences]


# ======================= PKUSEG =======================


@clock
def pkuseg_load():
    import pkuseg

    return pkuseg.pkuseg()


@clock
def pkuseg_cut(model, sentences):
    return [model.cut(sentence) for sentence in sentences]


# ======================= THULAC =======================
@clock
def thulac_load():
    import thulac

    thulac = thulac.thulac(seg_only=True)
    return thulac


@clock
def thulac_cut(model, sentences):
    return [model.cut(sentence, text=True) for sentence in sentences]


@clock
def thulac_fast_cut(model, sentences):
    return [model.fast_cut(sentence, text=True) for sentence in sentences]


# ======================= LTP 3 =======================


@clock
def ltp3_cws_load(model_path="../../data/legacy-models-v3/cws.model"):
    from pyltp import Segmentor

    model = Segmentor(model_path)
    return model


@clock
def ltp3_cut(model, sentences):
    return [list(model.segment(sentence)) for sentence in sentences]


@clock
def ltp3_load(
    cws_model_path="../../data/legacy-models-v3/cws.model",
    pos_model_path="../../data/legacy-models-v3/pos.model",
    ner_model_path="../../data/legacy-models-v3/ner.model",
):
    from pyltp import NamedEntityRecognizer, Postagger, Segmentor

    cws_model = Segmentor(cws_model_path)
    pos_model = Postagger(pos_model_path)
    ner_model = NamedEntityRecognizer(ner_model_path)
    return cws_model, pos_model, ner_model


@clock
def ltp3_pipeline(cws_model, pos_model, ner_model, sentences):
    batch_words = [cws_model.segment(sentence) for sentence in sentences]
    batch_pos = [pos_model.postag(words) for words in batch_words]
    batch_ner = [ner_model.recognize(words, pos) for (words, pos) in zip(batch_words, batch_pos)]

    return batch_words, batch_pos, batch_ner


# ======================= LTP =======================


@clock
def ltp_cws_load(model_path="../../data/legacy-models/cws_model.bin"):
    from ltp_extension.perceptron import CWSModel

    model = CWSModel.load(model_path)
    return model


@clock
def ltp_cut(model, sentences, threads=1):
    return model.batch_predict(sentences, threads)


@clock
def ltp_load(
    cws_model_path="../../data/legacy-models/cws_model.bin",
    pos_model_path="../../data/legacy-models/pos_model.bin",
    ner_model_path="../../data/legacy-models/ner_model.bin",
):
    from ltp_extension.perceptron import CWSModel, NERModel, POSModel

    cws_model = CWSModel.load(cws_model_path)
    pos_model = POSModel.load(pos_model_path)
    ner_model = NERModel.load(ner_model_path)
    return cws_model, pos_model, ner_model


@clock
def ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=1):
    batch_words = cws_model.batch_predict(sentences, threads)
    batch_pos = pos_model.batch_predict(batch_words, threads)
    batch_ner = ner_model.batch_predict(batch_words, batch_pos, threads)

    return batch_words, batch_pos, batch_ner


# ==================== Benchmark ====================


def cws_benchmark():
    with open("../../data/benchmark/pku-all.txt") as fi:
        sentences = [line.strip() for line in fi.readlines()]

    # 0.7831432819366455s
    jieba_model = jieba_load()
    # 37.776106119155884s
    jieba_cut(jieba_model, sentences)

    # 9.232479095458984s
    pkuseg = pkuseg_load()
    # 315.90687012672424s
    pkuseg_cut(pkuseg, sentences)

    # 0.9187619686126709s
    thulac_model = thulac_load()
    # 720.1864020824432s
    thulac_cut(thulac_model, sentences)
    # 30.58756685256958s
    thulac_fast_cut(thulac_model, sentences)

    # 0.17265582084655762s
    ltp3_model = ltp3_cws_load()
    # 76.8230140209198s
    ltp3_cut(ltp3_model, sentences)

    # 1.3583240509033203s
    ltp_model = ltp_cws_load()
    # 21.614962100982666s
    ltp_cut(ltp_model, sentences, threads=1)
    # 12.079923868179321s
    ltp_cut(ltp_model, sentences, threads=2)
    # 7.003376007080078s
    ltp_cut(ltp_model, sentences, threads=4)
    # 5.0945048332214355s
    ltp_cut(ltp_model, sentences, threads=8)
    # 4.475361108779907s
    ltp_cut(ltp_model, sentences, threads=16)


def pipeline_benchmark():
    with open("../../data/benchmark/pku-all.txt") as fi:
        sentences = [line.strip() for line in fi.readlines()]

    # 11.260580062866211s
    cws_model, pos_model, ner_model = ltp_load()

    # multi threads
    # 68.13384127616882s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=1)
    # 38.54593205451965s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=2)
    # 21.6906418800354s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=4)
    # 15.286723136901855s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=8)
    # 14.134387016296387s
    ltp_pipeline(cws_model, pos_model, ner_model, sentences, threads=16)

    # 0.8289380073547363s
    cws_model_3, pos_model_3, ner_model_3 = ltp3_load()
    # 226.40391397476196s
    ltp3_pipeline(cws_model_3, pos_model_3, ner_model_3, sentences)


def main():
    # cws_benchmark()
    pipeline_benchmark()


if __name__ == "__main__":
    main()
