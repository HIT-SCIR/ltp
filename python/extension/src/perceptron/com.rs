#[macro_export]
macro_rules! impl_model {
    ($name:ident) => {
        impl $name {
            fn inner_load(path: &str) -> anyhow::Result<Self> {
                use ltp::perceptron::ModelSerde;
                let file = std::fs::File::open(path)?;
                let model = if path.ends_with(".json") {
                    ModelSerde::load(file, ltp::perceptron::Format::JSON)?
                } else {
                    ModelSerde::load(
                        file,
                        ltp::perceptron::Format::AVRO(ltp::perceptron::Codec::Deflate),
                    )?
                };
                Ok(Self { model })
            }

            fn inner_save(&self, path: &str) -> anyhow::Result<()> {
                use ltp::perceptron::ModelSerde;
                let file = std::fs::File::create(path)?;
                if path.ends_with(".json") {
                    self.model.save(file, ltp::perceptron::Format::JSON)?;
                } else {
                    self.model.save(
                        file,
                        ltp::perceptron::Format::AVRO(ltp::perceptron::Codec::Deflate),
                    )?;
                }
                Ok(())
            }
        }
    };
    () => {};
}
