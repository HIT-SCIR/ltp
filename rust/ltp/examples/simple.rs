use itertools::multizip;
use ltp::{CWSModel, Codec, Format, ModelSerde, NERModel, POSModel};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data/legacy-models/cws_model.bin")?;
    let cws: CWSModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;
    let file = File::open("data/legacy-models/pos_model.bin")?;
    let pos: POSModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;
    let file = File::open("data/legacy-models/ner_model.bin")?;
    let ner: NERModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;

    let words = cws.predict("他叫汤姆去拿外衣。")?;
    let pos = pos.predict(&words)?;
    let ner = ner.predict((&words, &pos))?;

    for (w, p, n) in multizip((words, pos, ner)) {
        println!("{}/{}/{}", w, p, n);
    }

    Ok(())
}
