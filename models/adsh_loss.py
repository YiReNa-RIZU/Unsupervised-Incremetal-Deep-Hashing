import torch
import torch.nn as nn


class ADSH_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        loss = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * B.shape[0])

        return loss



class ADUH_Loss(nn.Module):
    """
    Loss function of ADUH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma):
        super(ADUH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        loss = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * B.shape[0])

        return loss
    
  
class UIDH_Loss(nn.Module):
    """
    Loss function of UIDH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma, mu, ita, loss_type):
        super(UIDH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma
        self.mu = mu
        self.ita = ita
        self.loss_type = loss_type

    def forward(self, F, F_old, old_F_data, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = ((F - B[omega, :]) ** 2).sum()
        correlation_loss = (F @ torch.ones(F.shape[1], 1, device=F.device)).sum()
        replace_loss = ((self.code_length - (F_old * old_F_data).sum(1)) ** 2).sum()
        if self.loss_type == 'uidh':
            loss = (hash_loss + self.gamma * quantization_loss + self.mu * correlation_loss) / (F.shape[0] * B.shape[0]) + self.ita * replace_loss / (F_old.shape[0])
        elif self.loss_type == 'adsh':
            loss = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * B.shape[0])
        else:
            raise ValueError('loss type error')
            
        return loss    