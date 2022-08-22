import functools
import os
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


# ======================= LTP =======================


@clock
def ltp_cws_load(model_path="../../data/legacy-models/cws_model.bin"):
    from ltp_extension.perceptron import CWSModel

    model = CWSModel.load(model_path)
    return model


@clock
def ltp_cut(model, sentences, threads=1):
    return model.batch_predict(sentences, threads)


# ==================== Benchmark ====================


def cws_benchmark(input_file, output_dir):
    with open(input_file) as fi:
        sentences = [line.strip() for line in fi.readlines()]
        sentences = [sentence for sentence in sentences if sentence]

    jieba_model = jieba_load()
    res = jieba_cut(jieba_model, sentences)
    with open(os.path.join(output_dir, "jieba.txt"), "w") as fo:
        for sentence in res:
            fo.write(" ".join(sentence) + "\n")

    pkuseg = pkuseg_load()
    res = pkuseg_cut(pkuseg, sentences)
    with open(os.path.join(output_dir, "pkuseg.txt"), "w") as fo:
        for sentence in res:
            fo.write(" ".join(sentence) + "\n")

    thulac_model = thulac_load()
    res = thulac_fast_cut(thulac_model, sentences)
    with open(os.path.join(output_dir, "thulac.txt"), "w") as fo:
        for sentence in res:
            fo.write(sentence + "\n")

    ltp3_model = ltp3_cws_load()
    res = ltp3_cut(ltp3_model, sentences)
    with open(os.path.join(output_dir, "ltp3.txt"), "w") as fo:
        for sentence in res:
            fo.write(" ".join(sentence) + "\n")

    ltp_model = ltp_cws_load()
    res = ltp_cut(ltp_model, sentences, threads=16)
    with open(os.path.join(output_dir, "ltp.txt"), "w") as fo:
        for sentence in res:
            fo.write(" ".join(sentence) + "\n")


def main():
    output_dir = "../../data/icwb2-data/predict/pku"
    os.makedirs(output_dir, exist_ok=True)
    cws_benchmark("../../data/icwb2-data/testing/pku_test.utf8", output_dir)

    output_dir = "../../data/icwb2-data/predict/msr"
    os.makedirs(output_dir, exist_ok=True)
    cws_benchmark("../../data/icwb2-data/testing/msr_test.utf8", output_dir)


if __name__ == "__main__":
    main()
