import torch
import torch.nn as nn

from plastic_transfer.core.encoders.base import Encoder

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors

        # surrogate gradient (sigmoid)
        x = input - threshold
        grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))

        return grad_output * grad, -grad_output * grad  # important
    



class LIFLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

        self.alpha_param = nn.Parameter(torch.ones(hidden_size) * 2.0)

        # THRESHOLD APRENDIBLE
        self.threshold = nn.Parameter(torch.ones(hidden_size) * 0.3)

    def forward(self, x):
        """
        x: [T, input_size]
        """
        T = x.shape[0]
        device = x.device

        v = torch.zeros(self.fc.out_features, device=device)
        spikes = []

        alpha = torch.sigmoid(self.alpha_param)

        for t in range(T):
            I = self.fc(x[t])

            v = alpha * v + I

            # s = SpikeFunction.apply(v, self.threshold)
            s = (v > self.threshold).float()

            v = v * (1 - s)  # reset

            spikes.append(s)

        spikes = torch.stack(spikes)  # [T, neurons]

        return spikes


class SNNEncoder(nn.Module, Encoder):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.lif1 = LIFLayer(input_size, hidden_size)
        self.lif2 = LIFLayer(hidden_size, latent_size)

        self.input_size = input_size

    def encode(self, trajectory):
        return self.forward(trajectory)
 
    def forward(self, trajectory):
        """
        trajectory: [T, input_size]
        return: spikes [T, latent_size]
        """
        x = trajectory

        spikes1 = self.lif1(x)
        spikes2 = self.lif2(spikes1)

        return spikes2