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


def main():
    issue590()
    issue592()


if __name__ == "__main__":
    main()
