from ltp import LTP


def issue590():
    ltp = LTP("LTP/tiny")
    ltp.add_words(words=["[ENT]"])
    print(ltp.pipeline(["[ENT] Info"], tasks=["cws"]))

    ltp.add_words(words=["[EOS]"])
    print(ltp.pipeline(["[EOS] Info"], tasks=["cws"]))


def issue592():
    legacy_ltp = LTP("LTP/legacy")

    legacy_ltp.add_words(words=["SCSG", "IP地址"])
    print(legacy_ltp.pipeline(["SCSGIP地址"], tasks=["cws"]))

    neural_ltp = LTP("LTP/tiny")

    # not bug, but not work because of the bert tokenizer
    neural_ltp.add_words(words=["SCSG", "IP地址"])
    print(neural_ltp.pipeline(["SCSGIP地址"], tasks=["cws"]))


def issue600():
    legacy_ltp = LTP("LTP/legacy")
    print(legacy_ltp.pipeline("他叫汤姆去拿外衣。", tasks=["cws"], return_dict=False))

    neural_ltp = LTP("LTP/tiny")
    print(neural_ltp.pipeline("他叫汤姆去拿外衣。", tasks=["cws"], return_dict=False))


def issue612():
    legacy_ltp = LTP("LTP/legacy")
    legacy_ltp.add_words(words=["五星武器"])
    print(legacy_ltp.pipeline("80 抽两五星武器给我吧哥", tasks=["cws"], return_dict=False))

    neural_ltp = LTP("LTP/tiny")
    neural_ltp.add_words(words=["五星武器"])
    print(neural_ltp.pipeline("80 抽两五星武器给我吧哥", tasks=["cws"], return_dict=False))


def issue613():
    import cProfile
    from pstats import SortKey

    cProfile.run('LTP("LTP/legacy", local_files_only=True)', sort=SortKey.CUMULATIVE)


from tqdm import trange
from matplotlib import pyplot as plt


def issue623():
    ltp = LTP("LTP/legacy")

    def get_current_memory() -> int:
        import os, psutil
        # 获取当前进程内存占用。
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        return info.uss / 1024 / 1024

    memory = [get_current_memory()]

    for _ in trange(10000):
        # ltp.pipeline('他叫汤姆去拿外衣。')
        # ltp.pipeline('台湾是中国领土不可分割的一部分。')
        ltp.pipeline(['他叫汤姆去拿外衣。', "台湾是中国领土不可分割的一部分。"])
        memory.append(get_current_memory())

    memory.append(get_current_memory())

    plt.plot(memory)
    plt.show()


def main():
    issue623()


if __name__ == "__main__":
    main()
