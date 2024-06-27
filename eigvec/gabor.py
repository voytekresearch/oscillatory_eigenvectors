"""Optimization Gabors."""

import torch
from torch import nn


class GaborLearner:
    """Optimization for torch model."""
    def __init__(self, window_length, n_gabors, lr=1e-2, l0_penalty=0.,
                 loss_fn=None, optimizer=None, n_epochs=1000,
                 normalize=True, verbose=None):

        self.window_length = window_length
        self.n_gabors = n_gabors
        self.l0_penalty = l0_penalty
        self.normalize = normalize
        self.verbose = verbose

        self.n_epochs = n_epochs
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.lr = lr
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer

        self.model_ = None


    def fit(self, X):

        # Reshape
        assert X.ndim == 2
        X = X.reshape(-1, self.window_length**2)

        # mean=0, variance=1
        if self.normalize:
            X = (X - X.mean()) / X.std()

        # Initialize base learner
        self.model_ = GaborModel(len(X), self.window_length, self.n_gabors)

        self.optimizer = self.optimizer(self.model_.parameters(), lr=self.lr)

        for i in range(self.n_epochs):

            # Forward
            X_pred = self.model_()

            if self.l0_penalty == 0.:
                # No penalty
                loss = self.loss_fn(X_pred, X)
            else:
                # L1 norm of weights
                #   weights are normalized to a unit vector prior to scaling
                penalty = self.l0_penalty * torch.norm(
                    self.model_.gabor_weights.view(-1) / \
                        torch.norm(self.model_.gabor_weights.view(-1), 2),
                    1
                )
                loss = self.loss_fn(X_pred, X) + penalty

            # Backward
            loss.backward()

            # Step optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.verbose is not None and i % self.verbose == 0:

                if i + self.verbose > self.n_epochs-1:
                    end=None
                else:
                    end = "\r"

                if self.l0_penalty != 0:
                    print(f"epoch {i}, loss: {float(loss)-float(penalty)}, penalty: {float(penalty)}",
                          end=end)
                else:
                    print(f"epoch {i}, loss: {float(loss)}", end=end)



class GaborModel(nn.Module):
    """Torch model."""
    def __init__(self, n_windows, size, n_gabors):

        super().__init__()

        self.size = size
        self.size_sq = int(size ** 2)
        ymin = xmin = -self.size//2
        ymax = xmax = self.size + ymin

        self.y, self.x = torch.meshgrid(
            torch.linspace(ymin, ymax, self.size),
            torch.linspace(xmin, xmax, self.size)
        )

        self.n_gabors = n_gabors

        self.x = self.x.reshape(*self.x.shape, 1)
        self.x = self.x.repeat((1, 1, self.n_gabors))

        self.y = self.y.reshape(*self.y.shape, 1)
        self.y = self.y.repeat((1, 1, self.n_gabors))

        self.gamma = nn.Parameter(torch.randn(self.n_gabors))
        self.sigma_x = nn.Parameter(torch.randn(self.n_gabors))

        self.theta = nn.Parameter(torch.randn(self.n_gabors))
        self.lam = nn.Parameter(torch.randn(self.n_gabors))
        self.psi = nn.Parameter(torch.randn(self.n_gabors))

        self.gabor_weights = nn.Parameter(torch.randn(n_windows, self.n_gabors))

    def forward(self):
        gabors = self.gabors()
        gabors = gabors.reshape(len(gabors), -1)
        return self.gabor_weights @ gabors

    def gabors(self):

        stheta = torch.sin(self.theta)
        ctheta = torch.cos(self.theta)

        x_theta = self.x * ctheta + self.y * stheta
        y_theta = -self.x * stheta + self.y * ctheta

        X = torch.exp(
            -0.5 * (x_theta**2 / self.sigma_x**2 + y_theta**2 / (self.sigma_x / self.gamma)**2)
        ) * torch.cos(2 * torch.pi / self.lam * x_theta + self.psi)

        X = X.permute(*torch.arange(X.ndim - 1, -1, -1))

        return X