# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LBPSemanticDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals of semantic dependency trees.
    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.
    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logQ = self.belief_propagation(*(s.requires_grad_() for s in scores), mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = -logQ.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def belief_propagation(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # log potentials of unary and binary factors
        # [2, seq_len, seq_len, batch_size], (h->m)
        p_edge = torch.stack((torch.zeros_like(s_edge), s_edge)).permute(0, 3, 2, 1)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        p_sib = s_sib.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        p_cop = s_cop.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        p_grd = s_grd.permute(2, 1, 3, 0)

        # log beliefs
        # [2, seq_len, seq_len, batch_size], (h->m)
        b = p_edge.new_zeros(2, seq_len, seq_len, batch_size)
        # log messages of siblings
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_sib = p_sib.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of co-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->c)
        m_cop = p_cop.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of grand-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->g)
        m_grd = p_grd.new_zeros(2, seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            b = b.log_softmax(0)
            # m(ik->ij) = logsumexp(b(ik) - m(ij->ik) + p(ij->ik)) - m(ik->)
            m = b.unsqueeze(3) - m_sib
            m_sib = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + p_sib)).logsumexp(0)))
            m_sib = m_sib.transpose(2, 3).log_softmax(0)
            m = b.unsqueeze(3) - m_cop
            m_cop = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + p_cop)).logsumexp(0)))
            m_cop = m_cop.transpose(2, 3).log_softmax(0)
            m = b.unsqueeze(3) - m_grd
            m_grd = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + p_grd)).logsumexp(0)))
            m_grd = m_grd.transpose(2, 3).log_softmax(0)
            # b(ij) = p(ij) + sum(m(ik->ij)), min(i, j) < k < max(i, j)
            b = p_edge + ((m_sib + m_cop + m_grd) * mask2o).sum(3)

        return b.permute(3, 2, 1, 0).log_softmax(-1)


class MFVISemanticDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals of semantic dependency trees.
    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.
    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logQ = self.mean_field_variational_inference(*(s.requires_grad_() for s in scores), mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = -logQ.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def mean_field_variational_inference(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_edge = s_edge.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0) * mask2o

        # posterior distributions
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_edge.new_zeros(2, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.softmax(0)
            # f(ij) = sum(q(ik)s^sib(ij,ik) + q(kj)s^cop(ij,kj) + q(jk)s^grd(ij,jk)), k != i,j
            f = (q[1].unsqueeze(1) * s_sib + q[1].transpose(0, 1).unsqueeze(0) * s_cop + q[1].unsqueeze(0) * s_grd).sum(
                2)
            # q(ij) = s(ij) + f(ij)
            q = torch.stack((torch.zeros_like(q[0]), s_edge + f))

        return q.permute(3, 2, 1, 0).log_softmax(-1)
