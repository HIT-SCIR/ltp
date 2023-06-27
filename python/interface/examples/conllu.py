#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>


from ltp import LTP


class Token:
    def __init__(self, id, form, lemma, upos, xpos, feats, head, deprel, deps, misc):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            self.id,
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats,
            self.head,
            self.deprel,
            self.deps,
            self.misc,
        )

    def __repr__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            self.id,
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats,
            self.head,
            self.deprel,
            self.deps,
            self.misc,
        )


def main():
    ltp = LTP("LTP/tiny")
    batched_cws, batched_pos, batched_dep, batched_sdpg = ltp.pipeline(
        ["他叫汤姆去拿外衣。", "他点头表示同意我的意见。", "我们即将以昂扬的斗志迎来新的一年。"], ["cws", "pos", "dep", "sdpg"]
    ).to_tuple()

    for cws, pos, dep, sdpg in zip(batched_cws, batched_pos, batched_dep, batched_sdpg):
        sentence = []
        for idx, (form, xpos, head, deprel) in enumerate(zip(cws, pos, dep["head"], dep["label"])):
            sentence.append(
                Token(
                    id=idx + 1,
                    form=form,
                    lemma="_",
                    upos="_",
                    xpos=xpos,
                    feats="_",
                    head=head,
                    deprel=deprel,
                    deps="",
                    misc="_",
                )
            )

        for id, head, tag in sdpg:
            if sentence[id - 1].deps:
                sentence[id - 1].deps = sentence[id - 1].deps + f"|{head}:{tag}"
            else:
                sentence[id - 1].deps = f"{head}:{tag}"

        sentence = [str(token) for token in sentence]
        sentence = "\n".join(sentence)

        print(sentence)
        print("\n")


if __name__ == "__main__":
    main()
