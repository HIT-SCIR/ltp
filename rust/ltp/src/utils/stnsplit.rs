#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitOptions {
    pub use_zh: bool,
    pub use_en: bool,
    pub bracket_as_entity: bool,
    pub zh_quote_as_entity: bool,
    pub en_quote_as_entity: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            use_zh: true,
            use_en: true,
            bracket_as_entity: true,
            zh_quote_as_entity: true,
            en_quote_as_entity: true,
        }
    }
}

pub fn stn_split(text: &str) -> Vec<&str> {
    let option = SplitOptions {
        use_zh: true,
        use_en: true,
        bracket_as_entity: true,
        zh_quote_as_entity: true,
        en_quote_as_entity: true,
    };
    stn_split_with_options(text, &option)
}

pub fn stn_split_with_options<'a, 'b>(text: &'a str, options: &'b SplitOptions) -> Vec<&'a str> {
    let mut res = vec![];
    let mut char_indices = text.char_indices().peekable();
    let mut quotes = SmallVec::<[char; 4]>::new();

    let mut next_start = 0usize;
    let mut as_normal = true;
    let mut post_quotes = false;
    let mut end_flag = false;
    let mut last_char = ' ';
    loop {
        if let Some((idx, ch)) = char_indices.next() {
            let next_char = char_indices.peek().map(|&(_, ch)| ch).unwrap_or('\n');
            match (last_char, next_char) {
                ('“', '”') | ('‘', '’') | ('『', '』') | ('﹃', '﹄') => {
                    if options.zh_quote_as_entity {
                        continue;
                    }
                }
                ('"', '"') | ('\'', '\'') => {
                    // quotes[..end] = [..lash_char];
                    if options.en_quote_as_entity && quotes.last() == Some(&next_char) {
                        continue;
                    }
                }
                _ => {} // do nothing
            }
            match ch {
                '\r' | '\n' => {
                    // skip empty sentence
                    if idx > next_start {
                        res.push(&text[next_start..idx]); // skip the '\n'
                    }
                    next_start = idx + 1;

                    // 换行则重置
                    end_flag = false;
                    quotes.clear();

                    // 不执行
                    as_normal = false;
                }
                '?' | '!' => {
                    // 英文符号
                    if options.use_en {
                        end_flag = true;
                        as_normal = false;
                    }
                }
                '"' | '\'' => {
                    // 英文引号
                    if options.en_quote_as_entity {
                        if quotes.is_empty() {
                            quotes.push(ch);
                        } else if quotes.last() == Some(&ch) {
                            quotes.pop();
                        } else {
                            quotes.push(ch);
                        }
                        as_normal = false;
                    }
                }
                '.' => {
                    if options.use_en {
                        if last_char.is_ascii_digit() && next_char.is_ascii_digit() {
                            // 小数点
                            end_flag = false;
                        } else {
                            // 英文句号
                            end_flag = true;
                        }
                        as_normal = false;
                    }
                }
                '！' | '？' | '。' => {
                    //  中文标点
                    if options.use_zh {
                        end_flag = true;
                        as_normal = false;
                    }
                }
                '…' | '⋯' | '᠁' => {
                    // 省略号不能作为开头
                    // 对于省略号啥也不干
                    if options.use_zh || options.use_en {
                        end_flag = true;
                        as_normal = false;
                    }
                }
                // 判断引号结束是不是应该成句，看引号中是否为完整句子。
                '“' | '‘' | '『' | '﹃' => {
                    if options.zh_quote_as_entity {
                        post_quotes = true;
                    }
                }
                '”' | '’' | '』' | '﹄' => {
                    if options.zh_quote_as_entity {
                        if let Some(quote) = quotes.pop() {
                            match (quote, ch) {
                                ('“', '”') | ('‘', '’') | ('『', '』') | ('﹃', '﹄') =>
                                    {
                                        // do nothing
                                    }
                                _ => {
                                    // todo: error: maybe need todo something ?
                                    // pop until find the quote pair
                                }
                            }
                        }
                        as_normal = false;
                    }
                }
                '[' | '(' | '{' | '⟨' | '（' | '〔' | '〈' | '《' | '【' => {
                    if options.bracket_as_entity {
                        // 左括号/左书名号 brackets
                        post_quotes = true;
                    }
                }
                ']' | ')' | '}' | '⟩' | '）' | '〕' | '〉' | '》' | '】' => {
                    // 右括号/右书名号
                    if options.bracket_as_entity {
                        if let Some(quote) = quotes.pop() {
                            match (quote, ch) {
                                ('[', ']')
                                | ('(', ')')
                                | ('{', '}')
                                | ('⟨', '⟩')
                                | ('（', '）')
                                | ('〔', '〕')
                                | ('〈', '〉')
                                | ('《', '》')
                                | ('【', '】') => {
                                    // do nothing
                                }
                                _ => {
                                    // todo: error: maybe need todo something ?
                                    // pop until find the brackets pair
                                }
                            }
                        }
                        as_normal = false;
                    }
                }
                _ => {}
            }
            if as_normal {
                if end_flag && quotes.is_empty() {
                    if idx > next_start {
                        res.push(&text[next_start..idx]);
                    }
                    next_start = idx;
                }
                // 比如 “全是东洋货！妈呀，”林小姐哭丧着脸说，“明儿叫我穿什么衣服？”
                // 引号之前的前半句不完整，不应当进行切分，但引号中间的其他句子可能已经修改了成句标记。
                end_flag = false;
            } else {
                as_normal = true;
            }

            if post_quotes {
                quotes.push(ch);
                post_quotes = false;
            }

            last_char = ch;
        } else {
            if next_start < text.len() {
                res.push(&text[next_start..]);
            }
            break;
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert!(stn_split("").is_empty());
        assert!(stn_split("\n").is_empty());
        assert!(stn_split("\n\n\n").is_empty());
    }

    #[test]
    fn test_common() {
        // common sentence
        assert_eq!(
            stn_split(concat!(
            "中国是世界上历史最悠久的国家之一。",
            "今天你怎么没给我打电话？",
            "还是历来惯了，不以为非呢？还是丧了良心，明知故犯呢？",
            "全国各民族大团结万岁！",
            "宁为玉碎，不为瓦全。她要揭露！要控诉!!要以死作最后的抗争!!!",
            "“什么？”男人强烈抗议道，“你以为我会随便退出娱乐圈吗?!”",
            "周朴园：鲁大海，你现在没有资格和我说话——矿上已经把你开除了。",
            "鲁大海：开除了?!",
            "要普及现代信息技术教育，“计算机要从娃娃抓起”。",
            "“坤包、坤表、坤车”里的“坤”，意思是女式的，女用的。",
            "《毛泽东选集》对“李林甫”是这样注释的：“李林甫，公元八世纪人，唐玄宗时的一个宰相。《资治通鉴》说：‘李林甫为相，凡才望功业出己右及为上所厚、势位将逼己者，必百计去之；尤忌文学之士，或阳与之善，啗以甘言而阴陷之。世谓李林甫“口有蜜，腹有剑”。’”",
            "大革命虽然失败了，但火种犹存。共产党人“从地下爬起来，揩干净身上的血迹，掩埋好同伴的尸首，他们又继续战斗了”。",
            )),
            vec![
                "中国是世界上历史最悠久的国家之一。",
                "今天你怎么没给我打电话？",
                "还是历来惯了，不以为非呢？", "还是丧了良心，明知故犯呢？",  // split it
                "全国各民族大团结万岁！",
                "宁为玉碎，不为瓦全。", "她要揭露！", "要控诉!!", "要以死作最后的抗争!!!",  // split it
                "“什么？”", "男人强烈抗议道，“你以为我会随便退出娱乐圈吗?!”",  // split it
                "周朴园：鲁大海，你现在没有资格和我说话——矿上已经把你开除了。",
                "鲁大海：开除了?!",
                "要普及现代信息技术教育，“计算机要从娃娃抓起”。",
                "“坤包、坤表、坤车”里的“坤”，意思是女式的，女用的。",
                "《毛泽东选集》对“李林甫”是这样注释的：“李林甫，公元八世纪人，唐玄宗时的一个宰相。《资治通鉴》说：‘李林甫为相，凡才望功业出己右及为上所厚、势位将逼己者，必百计去之；尤忌文学之士，或阳与之善，啗以甘言而阴陷之。世谓李林甫“口有蜜，腹有剑”。’”",
                "大革命虽然失败了，但火种犹存。", "共产党人“从地下爬起来，揩干净身上的血迹，掩埋好同伴的尸首，他们又继续战斗了”。",  // split it
            ]
        );
    }

    #[test]
    fn test_quoted() {
        // quoted sentence
        assert_eq!(
            stn_split(concat!(
            "丫姑折断几枝扔下来，边叫我的小名儿边说：“先喂饱你！”",
            "“哎呀，真是美极了，”皇帝说，“我十分满意！”",
            "从山脚向上望，只见火把排成许多“之”字形，一直连到天上。",
            "“怕什么！海的美就在这里！”我说。",
            "适当地改善自己的生活，岂但“你管得着吗”，而且是顺乎天理，合乎人情的。",
            "现代画家徐悲鸿笔下的马，正如有的评论家所说的那样，“形神兼备，充满生机”。",
            "唐朝的张嘉贞说它“制造奇特，人不知其所为”。",
            "我听见韩麦尔先生对我说：“唉，总要把学习拖到明天，这正是阿尔萨斯人最大的不幸。现在那些家伙就有理由对我们说了：‘怎么？你们还自己说是法国人呢，你们连自己的语言都不会说，不会写！……’不过，可怜的小弗郎士，也并不是你一个人的过错。”",
            "说他脸上有些妖气，一定遇见“美女蛇”了。",
            "他们（指友邦人士）维持他们“秩序”监狱，就撕掉了他们的“文明”的面具。",
            )),
            vec![
                "丫姑折断几枝扔下来，边叫我的小名儿边说：“先喂饱你！”",
                "“哎呀，真是美极了，”皇帝说，“我十分满意！”",
                "从山脚向上望，只见火把排成许多“之”字形，一直连到天上。",
                "“怕什么！海的美就在这里！”", "我说。",  // todo: maybe not split it
                "适当地改善自己的生活，岂但“你管得着吗”，而且是顺乎天理，合乎人情的。",
                "现代画家徐悲鸿笔下的马，正如有的评论家所说的那样，“形神兼备，充满生机”。",
                "唐朝的张嘉贞说它“制造奇特，人不知其所为”。",
                "我听见韩麦尔先生对我说：“唉，总要把学习拖到明天，这正是阿尔萨斯人最大的不幸。现在那些家伙就有理由对我们说了：‘怎么？你们还自己说是法国人呢，你们连自己的语言都不会说，不会写！……’不过，可怜的小弗郎士，也并不是你一个人的过错。”",
                "说他脸上有些妖气，一定遇见“美女蛇”了。",
                "他们（指友邦人士）维持他们“秩序”监狱，就撕掉了他们的“文明”的面具。",
            ]
        );
    }

    #[test]
    fn test_dotted() {
        // dotted sentence
        assert_eq!(
            stn_split(concat!(
            "那孩子含着泪唱着：“……世上只有妈妈好，没妈的孩子像根草……”",
            "各种鲜花争奇斗艳：菊花、玫瑰、马蹄莲、郁金香……",
            "他吃力地张开嘴：“你……要……坚持……下……去……”",
            )),
            vec![
                "那孩子含着泪唱着：“……世上只有妈妈好，没妈的孩子像根草……”",
                "各种鲜花争奇斗艳：菊花、玫瑰、马蹄莲、郁金香……",
                "他吃力地张开嘴：“你……要……坚持……下……去……”",
            ]
        );
    }

    #[test]
    fn test_special_quote() {
        // special quote sentence
        assert_eq!(
            stn_split(concat!(
            "林小姐哭丧着脸说：“妈呀，全是东洋货！明儿叫我穿什么衣服？”",
            "“妈呀，全是东洋货！明儿叫我穿什么衣服？”林小姐哭丧着脸说。",
            "“妈呀，”林小姐哭丧着脸说，“全是东洋货！明儿叫我穿什么衣服？”",
            "“全是东洋货！妈呀，”林小姐哭丧着脸说，“明儿叫我穿什么衣服？”",
            )),
            vec![
                "林小姐哭丧着脸说：“妈呀，全是东洋货！明儿叫我穿什么衣服？”",
                "“妈呀，全是东洋货！明儿叫我穿什么衣服？”",
                "林小姐哭丧着脸说。", // todo: maybe not split it
                "“妈呀，”林小姐哭丧着脸说，“全是东洋货！明儿叫我穿什么衣服？”",
                "“全是东洋货！妈呀，”林小姐哭丧着脸说，“明儿叫我穿什么衣服？”",
            ]
        );
    }

    #[test]
    fn test_mix() {
        // chinese & english & point number
        assert_eq!(
            stn_split(concat!(
            "中文和外文同时大量混排（如讲解英语语法的中文书），为避免中文小圆圈的句号“。”和西文小圆点儿的句号“.”穿插使用的不便，可以统统采用西文句号“.”。",
            "这个句子应当翻译成He loves sports.",
            "焦耳定律的公式是：Q = I2RT.",
            "计算所得的结果是48.2%.",
            "“‘”应该和“’”成对使用。",
            )),
            vec![
                "中文和外文同时大量混排（如讲解英语语法的中文书），为避免中文小圆圈的句号“。”和西文小圆点儿的句号“.”穿插使用的不便，可以统统采用西文句号“.”。",
                "这个句子应当翻译成He loves sports.",
                "焦耳定律的公式是：Q = I2RT.",
                "计算所得的结果是48.2%.",
                "“‘”应该和“’”成对使用。",
            ]
        );
    }
}
