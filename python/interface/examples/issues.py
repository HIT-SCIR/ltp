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


def main():
    issue613()


if __name__ == "__main__":
    main()
