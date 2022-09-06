try:
    import torch
    from ltp_extension.algorithms import eisner as rust_eisner
    from ltp_extension.algorithms import get_entities

    def eisner(scores, mask, remove_root=False):

        scores = scores.view(-1).cpu().numpy()
        length = torch.sum(mask, dim=1).cpu().numpy()

        result = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(sequence, device=mask.device)
                for sequence in rust_eisner(scores.tolist(), length.tolist(), remove_root)
            ],
            batch_first=True,
            padding_value=0,
        )

        return result

except Exception as e:
    from .eisner import eisner
    from .get_entities import get_entities
