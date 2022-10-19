from ltp import LTP
from ltp.legacy import CharacterType


def rules():
    ltp = LTP("LTP/legacy")
    result = ltp(["视频4k60fps无bg"], tasks=["cws"])
    print(result.cws)
    ltp.enable_type_cut_d(CharacterType.Roman, CharacterType.Kanji)
    ltp.enable_type_concat(CharacterType.Digit, CharacterType.Roman)
    result = ltp(["视频4k60fps无bg"], tasks=["cws"])
    print(result.cws)


def main():
    rules()


if __name__ == "__main__":
    main()
