def vocab_builder(func):
    from datasets import BuilderConfig

    def func_wrapper(config: BuilderConfig, **kwargs):
        """We handle string, list and dicts in datafiles."""
        if not config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={config.data_files}"
            )
        data_files = config.data_files
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
        else:
            files = []
            for file_list in data_files.values():
                if isinstance(file_list, str):
                    files.append(file_list)
                else:
                    files.extend(file_list)
        res = func(config.data_dir, *files, **kwargs)
        return res

    return func_wrapper
