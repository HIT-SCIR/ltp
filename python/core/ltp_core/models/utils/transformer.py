def load_transformers(config):
    from transformers import AutoModel, AutoConfig

    config = AutoConfig.for_model(**config)
    return AutoModel.from_config(config)
