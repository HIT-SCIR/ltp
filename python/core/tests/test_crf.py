import itertools
import math
import random

import pytest
import torch
import torch.nn as nn
from pytest import approx

from ltp_core.models.nn.crf import CRF

RANDOM_SEED = 1478754

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def compute_score(crf, emission, tag):
    # emission: (seq_length, num_tags)
    assert emission.dim() == 2
    assert emission.size(0) == len(tag)
    assert emission.size(1) == crf.num_tags
    assert all(0 <= t < crf.num_tags for t in tag)

    # Add transitions score
    score = crf.start_transitions[tag[0]] + crf.end_transitions[tag[-1]]
    for cur_tag, next_tag in zip(tag, tag[1:]):
        score += crf.transitions[cur_tag, next_tag]

    # Add emission score
    for emit, t in zip(emission, tag):
        score += emit[t]

    return score


def make_crf(num_tags=5, batch_first=False):
    return CRF(num_tags, batch_first=batch_first)


def make_emissions(crf, seq_length=3, batch_size=2):
    em = torch.randn(seq_length, batch_size, crf.num_tags)
    if crf.batch_first:
        em = em.transpose(0, 1)
    return em


def make_tags(crf, seq_length=3, batch_size=2):
    # shape: (seq_length, batch_size)
    ts = torch.tensor(
        [[random.randrange(crf.num_tags) for b in range(batch_size)] for _ in range(seq_length)],
        dtype=torch.long,
    )
    if crf.batch_first:
        ts = ts.transpose(0, 1)
    return ts


class TestInit:
    def test_minimal(self):
        num_tags = 10
        crf = CRF(num_tags)

        assert crf.num_tags == num_tags
        assert crf.batch_first
        assert isinstance(crf.start_transitions, nn.Parameter)
        assert crf.start_transitions.shape == (num_tags,)
        assert isinstance(crf.end_transitions, nn.Parameter)
        assert crf.end_transitions.shape == (num_tags,)
        assert isinstance(crf.transitions, nn.Parameter)
        assert crf.transitions.shape == (num_tags, num_tags)
        assert repr(crf) == f"CRF(num_tags={num_tags})"

    def test_full(self):
        crf = CRF(10, batch_first=False)
        assert not crf.batch_first

    def test_nonpositive_num_tags(self):
        with pytest.raises(ValueError) as excinfo:
            CRF(0)
        assert "invalid number of tags: 0" in str(excinfo.value)


class TestForward:
    def test_works_with_mask(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, seq_length, batch_size)
        # mask should have size of (seq_length, batch_size)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)

        # shape: ()
        llh = crf(emissions, tags, mask=mask)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.transpose(0, 1)
        # shape: (batch_size, seq_length)
        mask = mask.transpose(0, 1)

        # Compute log likelihood manually
        manual_llh = 0.0
        for emission, tag, mask_ in zip(emissions, tags, mask):
            seq_len = mask_.sum()
            emission, tag = emission[:seq_len], tag[:seq_len]
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_len)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator

        assert llh.detach().cpu().item() == approx(manual_llh.detach().cpu().item())
        llh.backward()  # ensure gradients can be computed

    def test_works_without_mask(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        llh_no_mask = crf(emissions, tags)
        # No mask means the mask is all ones
        llh_mask = crf(emissions, tags, mask=torch.ones_like(tags).byte())

        assert llh_no_mask.detach().cpu().item() == approx(llh_mask.detach().cpu().item())

    def test_batched_loss(self):
        crf = make_crf()
        batch_size = 10

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, batch_size=batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, batch_size=batch_size)

        llh = crf(emissions, tags)
        assert torch.is_tensor(llh)
        assert llh.shape == ()

        total_llh = 0.0
        for i in range(batch_size):
            # shape: (seq_length, 1, num_tags)
            emissions_ = emissions[:, i, :].unsqueeze(1)
            # shape: (seq_length, 1)
            tags_ = tags[:, i].unsqueeze(1)
            # shape: ()
            total_llh += crf(emissions_, tags_)

        assert llh.detach().cpu().item() == approx(total_llh.detach().cpu().item())

    def test_reduction_none(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        seq_length, batch_size = tags.shape

        llh = crf(emissions, tags, reduction="none")

        assert torch.is_tensor(llh)
        assert llh.shape == (batch_size,)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.transpose(0, 1)

        # Compute log likelihood manually
        manual_llh = []
        for emission, tag in zip(emissions, tags):
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_length)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh.append(numerator - denominator)

        for llh_, manual_llh_ in zip(llh, manual_llh):
            assert llh_.detach().cpu().item() == approx(manual_llh_.detach().cpu().item())

    def test_reduction_mean(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)

        seq_length, batch_size = tags.shape

        llh = crf(emissions, tags, reduction="mean")

        assert torch.is_tensor(llh)
        assert llh.shape == ()

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.transpose(0, 1)

        # Compute log likelihood manually
        manual_llh = 0
        for emission, tag in zip(emissions, tags):
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_length)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator

        assert llh.detach().cpu().item() == approx(manual_llh.detach().cpu().item() / batch_size)

    def test_reduction_token_mean(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, seq_length, batch_size)
        # mask should have size of (seq_length, batch_size)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)

        llh = crf(emissions, tags, mask=mask, reduction="token_mean")

        assert torch.is_tensor(llh)
        assert llh.shape == ()

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.transpose(0, 1)
        # shape: (batch_size, seq_length)
        mask = mask.transpose(0, 1)

        # Compute log likelihood manually
        manual_llh, n_tokens = 0, 0
        for emission, tag, mask_ in zip(emissions, tags, mask):
            seq_len = mask_.sum()
            emission, tag = emission[:seq_len], tag[:seq_len]
            numerator = compute_score(crf, emission, tag)
            all_scores = [
                compute_score(crf, emission, t)
                for t in itertools.product(range(crf.num_tags), repeat=seq_len)
            ]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator
            n_tokens += seq_len

        assert llh.detach().cpu().item() == approx(manual_llh.detach().cpu().item() / n_tokens)

    def test_batch_first(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)
        llh = crf(emissions, tags)

        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.requires_grad_(False).copy_(crf.start_transitions)
        crf_bf.end_transitions.requires_grad_(False).copy_(crf.end_transitions)
        crf_bf.transitions.requires_grad_(False).copy_(crf.transitions)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.transpose(0, 1)
        llh_bf = crf_bf(emissions, tags)

        assert llh.item() == approx(llh_bf.item())

    def test_scripted_forward(self):
        # Test default case
        crf = make_crf()
        crf_script = torch.jit.script(crf)
        seq_length, batch_size = 3, 2
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf, seq_length, batch_size)
        # mask should have size of (seq_length, batch_size)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)
        llh = crf(emissions, tags, mask=mask)
        llh_scripted = crf_script(emissions, tags, mask=mask)
        assert torch.equal(llh, llh_scripted), (
            f"scripted crf forward output {llh_scripted} "
            f"not matching non-scripted forward output {llh}"
        )

        # Test scripted forward works without mask
        llh_no_mask = crf(emissions, tags)
        llh_no_mask_script = crf_script(emissions, tags)
        assert torch.equal(llh_no_mask, llh_no_mask_script), (
            f"scripted crf forward output {llh_no_mask_script} "
            f"not matching non-scripted forward output {llh_no_mask}"
        )
        # No mask means the mask is all ones
        llh_mask = crf(emissions, tags, mask=torch.ones_like(tags).byte())
        llh_mask_script = crf_script(emissions, tags, mask=torch.ones_like(tags).byte())
        assert torch.equal(llh_mask, llh_mask_script), (
            f"scripted crf forward output {llh_mask_script} "
            f"not matching non-scripted forward output {llh_mask}"
        )

        # Test scripted forward in batched setting
        batch_size = 10
        # shape: (seq_length, batch_size, num_tags)
        emissions_batch = make_emissions(crf, batch_size=batch_size)
        # shape: (seq_length, batch_size)
        tags_batch = make_tags(crf, batch_size=batch_size)
        llh = crf(emissions_batch, tags_batch)
        llh_script = crf_script(emissions_batch, tags_batch)
        assert torch.equal(llh_script, llh), (
            f"scripted crf forward output {llh_script} "
            f"not matching non-scripted forward output {llh}"
        )

        # Test scripted forward when reduction is None, mean, token_mean
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        # shape: (seq_length, batch_size)
        tags = make_tags(crf)
        llh = crf(emissions, tags, reduction="none")
        llh_script = crf_script(emissions, tags, reduction="none")
        assert torch.equal(llh_script, llh), (
            f"scripted crf forward output {llh_script} "
            f"not matching non-scripted forward output {llh}"
        )
        llh = crf(emissions, tags, reduction="mean")
        llh_script = crf_script(emissions, tags, reduction="mean")
        assert torch.equal(llh_script, llh), (
            f"scripted crf forward output {llh_script} "
            f"not matching non-scripted forward output {llh}"
        )

        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)
        llh = crf(emissions, tags, mask=mask, reduction="token_mean")
        llh_script = crf_script(emissions, tags, mask=mask, reduction="token_mean")
        assert torch.equal(llh_script, llh), (
            f"scripted crf forward output {llh_script} "
            f"not matching non-scripted forward output {llh}"
        )

        # Test scripted forward when running batch first mode
        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.requires_grad_(False).copy_(crf.start_transitions)
        crf_bf.end_transitions.requires_grad_(False).copy_(crf.end_transitions)
        crf_bf.transitions.requires_grad_(False).copy_(crf.transitions)
        crf_bf_script = torch.jit.script(crf_bf)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        tags = tags.transpose(0, 1)
        llh_bf = crf_bf(emissions, tags)
        llh_bf_script = crf_bf_script(emissions, tags)
        assert torch.equal(llh_bf_script, llh_bf), (
            f"scripted crf forward output {llh_bf_script} "
            f"not matching non-scripted forward output {llh_bf}"
        )

    def test_emissions_has_bad_number_of_dimension(self):
        emissions = torch.randn(1, 2)
        tags = torch.empty(2, 2, dtype=torch.long)
        crf = make_crf()

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert "emissions must have dimension of 3, got 2" in str(excinfo.value)

    def test_emissions_and_tags_size_mismatch(self):
        emissions = torch.randn(1, 2, 3)
        tags = torch.empty(2, 2, dtype=torch.long)
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert (
            "the first two dimensions of emissions and tags must match, " "got (1, 2) and (2, 2)"
        ) in str(excinfo.value)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        emissions = torch.randn(1, 2, 3)
        tags = torch.empty(1, 2, dtype=torch.long)
        crf = make_crf(10)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert "expected last dimension of emissions is 10, got 3" in str(excinfo.value)

    def test_first_timestep_mask_is_not_all_on(self):
        emissions = torch.randn(3, 2, 4)
        tags = torch.empty(3, 2, dtype=torch.long)
        mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.uint8).transpose(0, 1)
        crf = make_crf(4)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, mask=mask)
        assert "mask of the first timestep must all be on" in str(excinfo.value)

        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)
        crf = make_crf(4, batch_first=True)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, mask=mask)
        assert "mask of the first timestep must all be on" in str(excinfo.value)

    def test_invalid_reduction(self):
        crf = make_crf()
        emissions = make_emissions(crf)
        tags = make_tags(crf)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, reduction="foo")
        assert "invalid reduction: foo" in str(excinfo.value)


class TestDecode:
    def test_works_with_mask(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # mask should be (seq_length, batch_size)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)

        best_tags = crf.decode(emissions, mask=mask)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        # shape: (batch_size, seq_length)
        mask = mask.transpose(0, 1)

        # Compute best tag manually
        for emission, best_tag, mask_ in zip(emissions, best_tags, mask):
            seq_len = mask_.sum()
            assert len(best_tag) == seq_len
            assert all(isinstance(t, int) for t in best_tag)
            emission = emission[:seq_len]
            manual_best_tag = max(
                itertools.product(range(crf.num_tags), repeat=seq_len),
                key=lambda t: compute_score(crf, emission, t),
            )
            assert tuple(best_tag) == manual_best_tag

    def test_works_without_mask(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)

        best_tags_no_mask = crf.decode(emissions)
        # No mask means mask is all ones
        best_tags_mask = crf.decode(emissions, mask=emissions.new_ones(emissions.shape[:2]).byte())

        assert best_tags_no_mask == best_tags_mask

    def test_batched_decode(self):
        crf = make_crf()
        batch_size, seq_length = 2, 3

        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)

        batched = crf.decode(emissions, mask=mask)

        non_batched = []
        for i in range(batch_size):
            # shape: (seq_length, 1, num_tags)
            emissions_ = emissions[:, i, :].unsqueeze(1)
            # shape: (seq_length, 1)
            mask_ = mask[:, i].unsqueeze(1)

            result = crf.decode(emissions_, mask=mask_)
            assert len(result) == 1
            non_batched.append(result[0])

        assert non_batched == batched

    def test_batch_first(self):
        crf = make_crf()
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf)
        best_tags = crf.decode(emissions)

        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.requires_grad_(False).copy_(crf.start_transitions)
        crf_bf.end_transitions.requires_grad_(False).copy_(crf.end_transitions)
        crf_bf.transitions.requires_grad_(False).copy_(crf.transitions)

        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        best_tags_bf = crf_bf.decode(emissions)

        assert best_tags == best_tags_bf

    def test_scripted_decode(self):
        # Test decoding with a mask
        crf = make_crf()
        crf_script = torch.jit.script(crf)

        seq_length, batch_size = 3, 2
        # shape: (seq_length, batch_size, num_tags)
        emissions = make_emissions(crf, seq_length, batch_size)
        # mask should be (seq_length, batch_size)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)
        best_tags = crf.decode(emissions, mask=mask)
        best_tags_scripted = crf_script.decode(emissions, mask=mask)
        assert best_tags == best_tags_scripted, (
            f"scripted decode output {best_tags_scripted} "
            f"doesn't match non-scripted output {best_tags}"
        )

        # Test decoding without a mask
        best_tags_no_mask = crf.decode(emissions)
        best_tags_no_mask_scripted = crf_script.decode(emissions)
        assert best_tags_no_mask == best_tags_no_mask_scripted, (
            f"scripted decode output {best_tags_no_mask_scripted} "
            f"doesn't match non-scripted output {best_tags_no_mask}"
        )

        # Test batched decode
        batch_size, seq_length = 2, 3
        # shape: (seq_length, batch_size, num_tags)
        emissions_batched = make_emissions(crf, seq_length, batch_size)
        # shape: (seq_length, batch_size)
        mask_batched = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8).transpose(0, 1)
        batched = crf.decode(emissions_batched, mask=mask_batched)
        batched_scripted = crf_script.decode(emissions_batched, mask=mask_batched)
        assert batched == batched_scripted, (
            f"scripted decode output {batched_scripted} "
            f"doesn't match non-scripted output {batched}"
        )

        # Test batch first decode
        crf_bf = make_crf(batch_first=True)
        # Copy parameter values from non-batch-first CRF; requires_grad must be False
        # to avoid runtime error of in-place operation on a leaf variable
        crf_bf.start_transitions.requires_grad_(False).copy_(crf.start_transitions)
        crf_bf.end_transitions.requires_grad_(False).copy_(crf.end_transitions)
        crf_bf.transitions.requires_grad_(False).copy_(crf.transitions)
        crf_bf_script = torch.jit.script(crf_bf)
        # shape: (batch_size, seq_length, num_tags)
        emissions = emissions.transpose(0, 1)
        best_tags_bf = crf_bf.decode(emissions)
        best_tags_bf_script = crf_bf_script.decode(emissions)
        assert best_tags_bf == best_tags_bf_script, (
            f"scripted decode output {best_tags_bf_script} "
            f"doesn't match non-scripted decode output {best_tags_bf}"
        )

    def test_emissions_has_bad_number_of_dimension(self):
        emissions = torch.randn(1, 2)
        crf = make_crf()

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions)
        assert "emissions must have dimension of 3, got 2" in str(excinfo.value)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        emissions = torch.randn(1, 2, 3)
        crf = make_crf(10)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions)
        assert "expected last dimension of emissions is 10, got 3" in str(excinfo.value)

    def test_emissions_and_mask_size_mismatch(self):
        emissions = torch.randn(1, 2, 3)
        mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.uint8)
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert (
            "the first two dimensions of emissions and mask must match, " "got (1, 2) and (2, 2)"
        ) in str(excinfo.value)

    def test_first_timestep_mask_is_not_all_on(self):
        emissions = torch.randn(3, 2, 4)
        mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.uint8).transpose(0, 1)
        crf = make_crf(4)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert "mask of the first timestep must all be on" in str(excinfo.value)

        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        crf = make_crf(4, batch_first=True)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert "mask of the first timestep must all be on" in str(excinfo.value)
