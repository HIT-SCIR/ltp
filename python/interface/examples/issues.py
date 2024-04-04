def issue590():
    from ltp import LTP
    ltp = LTP("LTP/tiny")
    ltp.add_words(words=["[ENT]"])
    print(ltp.pipeline(["[ENT] Info"], tasks=["cws"]))

    ltp.add_words(words=["[EOS]"])
    print(ltp.pipeline(["[EOS] Info"], tasks=["cws"]))


def issue592():
    from ltp import LTP
    legacy_ltp = LTP("LTP/legacy")

    legacy_ltp.add_words(words=["SCSG", "IP地址"])
    print(legacy_ltp.pipeline(["SCSGIP地址"], tasks=["cws"]))

    neural_ltp = LTP("LTP/tiny")

    # not bug, but not work because of the bert tokenizer
    neural_ltp.add_words(words=["SCSG", "IP地址"])
    print(neural_ltp.pipeline(["SCSGIP地址"], tasks=["cws"]))


def issue600():
    from ltp import LTP
    legacy_ltp = LTP("LTP/legacy")
    print(legacy_ltp.pipeline("他叫汤姆去拿外衣。", tasks=["cws"], return_dict=False))

    neural_ltp = LTP("LTP/tiny")
    print(neural_ltp.pipeline("他叫汤姆去拿外衣。", tasks=["cws"], return_dict=False))


def issue612():
    from ltp import LTP
    legacy_ltp = LTP("LTP/legacy")
    legacy_ltp.add_words(words=["五星武器"])
    print(legacy_ltp.pipeline("80 抽两五星武器给我吧哥", tasks=["cws"], return_dict=False))

    neural_ltp = LTP("LTP/tiny")
    neural_ltp.add_words(words=["五星武器"])
    print(neural_ltp.pipeline("80 抽两五星武器给我吧哥", tasks=["cws"], return_dict=False))


def issue613():
    import cProfile
    from pstats import SortKey

    cProfile.run('from ltp import LTP;LTP("LTP/legacy", local_files_only=True)', sort=SortKey.CUMULATIVE)


def issue623():
    from ltp import LTP
    from matplotlib import pyplot as plt
    from tqdm import trange
    ltp = LTP("LTP/legacy")

    def get_current_memory() -> int:
        import os

        import psutil

        # 获取当前进程内存占用。
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        return info.uss / 1024 / 1024

    memory = [get_current_memory()]

    for _ in trange(10000):
        # ltp.pipeline('他叫汤姆去拿外衣。')
        # ltp.pipeline('台湾是中国领土不可分割的一部分。')
        ltp.pipeline(["他叫汤姆去拿外衣。", "台湾是中国领土不可分割的一部分。"])
        memory.append(get_current_memory())

    memory.append(get_current_memory())

    plt.plot(memory)
    plt.show()


def issue686():
    from ltp_extension.algorithms import Hook
    sentence = b'\xc2\x28'.decode('utf-8', 'replace')
    hook = Hook()
    hook.add_word(word="[FAKE]")
    try:
        hook.hook(sentence, ['a', 'b'])
    except Exception as e:
        print(e)


def main():
    issue686()


if __name__ == "__main__":
    main()
