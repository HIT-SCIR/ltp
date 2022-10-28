import torch
from ltp import LTP


def stn_split():
    from ltp import StnSplit
    sents = StnSplit().split("汤姆生病了。他去了医院。")
    print(sents)
    # [
    #   "汤姆生病了。",
    #   "他去了医院。"
    # ]

    sents = StnSplit().batch_split(["他叫汤姆去拿外衣。", "汤姆生病了。他去了医院。"])
    print(sents)
    # [
    #   "他叫汤姆去拿外衣。",
    #   "汤姆生病了。",
    #   "他去了医院。"
    # ]


def legacy():
    ltp = LTP("LTP/legacy")
    ltp.add_word("汤姆去")
    result = ltp(
        ["他叫汤姆去拿外衣。", "树上停着一些小鸟。先飞走了19只，又飞走了15只。两次共飞走了多少只小鸟？"],
        tasks=["cws", "pos", "ner"],
    )
    print(result.cws)
    print(result.pos)
    print(result.ner)


def neural():
    ltp = LTP("LTP/tiny")

    if torch.cuda.is_available():
        ltp = ltp.to("cuda")

    ltp.add_word("汤姆去")

    # 未分词的文本
    result = ltp.pipeline(
        ["他叫汤姆去拿外衣。", "韓語：한국의 단오", "树上停着一些小鸟。先飞走了19只，又飞走了15只。两次共飞走了多少只小鸟？"],
        tasks=["cws", "pos", "ner", "srl", "dep", "sdp"],
    )
    print(result.cws)
    print(result.pos)
    print(result.ner)
    print(result.srl)
    print(result.dep)
    print(result.sdp)

    # 已经分词的文本
    result = ltp.pipeline(
        [["他", "叫", "汤姆", "去", "拿", "外衣", "。"], ["가을동", "叫", "1993", "年", "的", "Ameri", "·"]],
        # 注意这里移除了 "cws" 任务
        tasks=["pos", "ner", "srl", "dep", "sdp"],
    )
    print(result.pos)
    print(result.ner)
    print(result.srl)
    print(result.dep)
    print(result.sdp)


def main():
    stn_split()
    legacy()
    neural()


if __name__ == "__main__":
    main()
