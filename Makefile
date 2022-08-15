
help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

sync: ## Merge changes from main branch to your current branch
	git fetch --all
	git merge main

bdist: ## build ltp and ltp_extension
	pip wheel --no-deps -w dist python/core
	pip wheel --no-deps -w dist python/interface
	maturin build --release -m python/extension/Cargo.toml --out dist

cbindgen_header:
	mkdir -p bindings/c
	cbindgen --config rust/ltp-cffi/cbindgen.toml --crate ltp-cffi --output bindings/c/ltp.h

cbindgen: cbindgen_header
	cargo build --release  --package ltp-cffi
	cp target/release/libltp.* bindings/c

cbindgen_example: cbindgen
	gcc -L "$(pwd)bindings/c" -lltp -I "$(pwd)bindings/c" -o target/c_example rust/ltp-cffi/examples/example.c
	./target/c_example

train_legacy:
	cargo run --package ltp --release --example cws -- train --train data/examples/cws/train.txt --eval data/examples/cws/val.txt --model=data/cws_model.bin
	cargo run --package ltp --release --example cws -- eval --eval data/examples/cws/test.txt --model=data/cws_model.bin
	cargo run --package ltp --release --example cws -- predict --input data/examples/cws/raw.txt --output data/examples/cws/output.txt --model=data/cws_model.bin

	cargo run --package ltp --release --example pos -- train --train data/examples/pos/train.txt --eval data/examples/pos/val.txt --model=data/pos_model.bin --vocab data/examples/pos/vocab.txt
	cargo run --package ltp --release --example pos -- eval --eval data/examples/pos/test.txt --model=data/pos_model.bin
	cargo run --package ltp --release --example pos -- predict --input data/examples/pos/raw.txt --output data/examples/pos/output.txt --model=data/pos_model.bin

	cargo run --package ltp --release --example ner -- train --train data/examples/ner/train.txt --eval data/examples/ner/val.txt --model=data/ner_model.bin --vocab data/examples/ner/vocab.txt
	cargo run --package ltp --release --example ner -- eval --eval data/examples/ner/test.txt --model=data/ner_model.bin
	cargo run --package ltp --release --example ner -- predict --input data/examples/ner/raw.txt --output data/examples/ner/output.txt --model=data/ner_model.bin
