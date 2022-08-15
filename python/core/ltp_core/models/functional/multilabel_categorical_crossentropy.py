import torch

# ref: https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py

INF = 1e4
EPSILON = 1e-5


def multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素非0即1，
           1表示对应的类为目标类，0表示对应的类为非目标类；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_neg = y_pred - y_true * INF
    y_pos = y_pred - (1 - y_true) * INF
    zeros = torch.zeros_like(y_pred[..., :1])
    y_neg = torch.cat([y_neg, zeros], dim=-1)
    y_pos = torch.cat([y_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)
    return pos_loss + neg_loss


def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)

    if mask_zero:
        infs = zeros + INF
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)

    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)

    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=EPSILON, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss
