def load_transformers(config):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(**config)
    return AutoModel.from_config(config)
