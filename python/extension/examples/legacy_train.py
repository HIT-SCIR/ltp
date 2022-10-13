from ltp_extension.perceptron import Algorithm, CWSModel, CWSTrainer, Model, ModelType, Trainer


def train_cws():
    ap = Algorithm("AP")
    pa = Algorithm("Pa")
    pai = Algorithm("PaI", 0.5)
    paii = Algorithm("PaII", 0.5)

    trainer: CWSTrainer = CWSTrainer()
    trainer.epoch = 10
    trainer.load_train_data("data/cws/val.txt")
    trainer.load_eval_data("data/cws/test.txt")
    print(trainer)

    for algorithm in [ap, pa, pai, paii]:
        print(algorithm)
        trainer.algorithm = algorithm
        _: CWSModel = trainer.train()


def train_auto():
    ap = Algorithm("AP")
    pa = Algorithm("Pa")
    pai = Algorithm("PaI", 0.5)
    paii = Algorithm("PaII", 0.5)

    model_type = ModelType("cws")
    trainer: Trainer = Trainer(model_type)
    trainer.epoch = 10
    trainer.load_train_data("data/cws/val.txt")
    trainer.load_eval_data("data/cws/test.txt")
    print(trainer)

    for algorithm in [ap, pa, pai, paii]:
        print(algorithm)
        trainer.algorithm = algorithm
        _: CWSModel = trainer.train()


def main():
    # train_cws()
    train_auto()


if __name__ == "__main__":
    main()
