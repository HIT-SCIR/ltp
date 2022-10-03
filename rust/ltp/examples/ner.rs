use anyhow::Result;
use clap::{Parser, ValueEnum};
use itertools::Itertools;
use ltp::perceptron::SerdeNERModel;
use ltp::{Algorithm, Codec, Format, ModelSerde, NERDefinition as Definition, PaMode, Trainer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
enum Args {
    Train(Train),
    Eval(Eval),
    Predict(Predict),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum AlgorithmArg {
    Ap,
    Pa,
    PaI,
    PaII,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Train {
    #[clap(long, value_parser, default_value_t = 10)]
    epoch: usize,
    #[clap(short, long, value_parser, default_value_t = true)]
    shuffle: bool,
    #[clap(value_enum, value_parser, default_value_t = AlgorithmArg::Ap)]
    algorithm: AlgorithmArg,
    #[clap(long, value_parser, default_value_t = 8)]
    ap_threads: usize,
    #[clap(long, value_parser, default_value_t = 0.5)]
    pa_margin: f64,

    #[clap(long, value_parser, default_value_t = 8)]
    threads: usize,
    #[clap(long, value_parser, default_value_t = 8)]
    eval_threads: usize,

    // 模型压缩参数
    #[clap(short, long, value_parser, default_value_t = true)]
    compress: bool,
    #[clap(long, value_parser, default_value_t = 0.3)]
    ratio: f64,
    #[clap(long, value_parser, default_value_t = 1e-3)]
    threshold: f64,

    // 数据集
    #[clap(short, long)]
    train: String,
    #[clap(short, long)]
    eval: String,

    // 模型保存
    #[clap(short, long)]
    vocab: String,

    #[clap(short, long, default_value = "model.bin")]
    model: String,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Eval {
    #[clap(short, long, value_parser, default_value_t = true)]
    verbose: bool,

    #[clap(short, long, value_parser, default_value_t = 8)]
    threads: usize,
    #[clap(short, long)]
    eval: String,

    // 模型保存
    #[clap(short, long, default_value = "model.bin")]
    model: String,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Predict {
    #[clap(short, long, value_parser, default_value_t = true)]
    verbose: bool,
    #[clap(short, long, value_parser, default_value_t = 8)]
    threads: usize,

    #[clap(short, long)]
    input: String,
    #[clap(short, long)]
    output: String,

    // 模型保存
    #[clap(short, long, default_value = "model.bin")]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args {
        Args::Train(mode) => {
            let vocab = File::open(&mode.vocab)?;
            let vocab = BufReader::new(vocab);
            let vocab = vocab.lines().flatten().collect();

            let algorithm = match mode.algorithm {
                AlgorithmArg::Ap => Algorithm::AP(mode.ap_threads),
                AlgorithmArg::Pa => Algorithm::PA(PaMode::Pa),
                AlgorithmArg::PaI => Algorithm::PA(PaMode::PaI(mode.pa_margin)),
                AlgorithmArg::PaII => Algorithm::PA(PaMode::PaII(mode.pa_margin)),
            };

            let trainer = Trainer::new()
                .definition(Definition::new(vocab))
                .epoch(mode.epoch)
                .algorithm(algorithm)
                .shuffle(mode.shuffle)
                .compress(mode.compress)
                .ratio(mode.ratio)
                .threshold(mode.threshold)
                .train_file(mode.train)?
                .eval_file(mode.eval)?
                .display();

            let model = trainer.build::<HashMap<String, usize>, Vec<_>>()?;

            let file = File::create(&mode.model)?;
            let format = if mode.model.ends_with(".json") {
                Format::JSON
            } else {
                Format::AVRO(Codec::Deflate)
            };
            model.save(file, format)?;
        }
        Args::Eval(mode) => {
            let file = File::open(&mode.model)?;
            let format = if mode.model.ends_with(".json") {
                Format::JSON
            } else {
                Format::AVRO(Codec::Deflate)
            };
            let model: SerdeNERModel = ModelSerde::load(file, format)?;

            let start = std::time::Instant::now();
            let trainer = Trainer::new()
                .definition(model.definition.clone())
                .verbose(mode.verbose)
                .eval_threads(mode.threads)
                .eval_file(mode.eval)?;

            let (p, r, f1) = trainer.evaluate(&model)?;
            let duration = start.elapsed().as_millis();
            println!("[{duration}ms] precision: {p}, recall: {r}, f1: {f1}",);
        }
        Args::Predict(mode) => {
            let file = File::open(&mode.model)?;
            let format = if mode.model.ends_with(".json") {
                Format::JSON
            } else {
                Format::AVRO(Codec::Deflate)
            };
            let model: SerdeNERModel = ModelSerde::load(file, format)?;
            let file = File::open(mode.input)?;
            let lines = BufReader::new(file).lines();

            let datasets = lines.flatten().filter(|s| !s.is_empty()).collect_vec();
            let mut all_words: Vec<Vec<&str>> = Vec::with_capacity(datasets.len());
            let mut all_pos: Vec<Vec<&str>> = Vec::with_capacity(datasets.len());
            for sentence in &datasets {
                let words_tags = sentence.split_whitespace().collect_vec();
                let mut sentence_words = Vec::with_capacity(words_tags.len());
                let mut sentence_pos = Vec::with_capacity(words_tags.len());
                for word_tag in words_tags {
                    let result = word_tag.rsplitn(2, '/');
                    let (pos, word) = result.collect_tuple().expect("tag not found");
                    sentence_words.push(word);
                    sentence_pos.push(pos);
                }
                all_words.push(sentence_words);
                all_pos.push(sentence_pos);
            }
            let start = std::time::Instant::now();
            let result: Result<Vec<Vec<&str>>> = all_words
                .into_iter()
                .zip(all_pos.into_iter())
                .map(|(words, pos)| model.predict((&words, &pos)))
                .collect();
            let duration = start.elapsed();
            println!("{}ms", duration.as_millis());
            let mut file = File::create(mode.output)?;
            result?.iter().for_each(|sentence| {
                writeln!(file, "{}", sentence.join(" ")).expect("Write Failed!");
            });
        }
    }

    Ok(())
}
