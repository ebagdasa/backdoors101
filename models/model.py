import torch.nn as nn


class Model(nn.Module):
    """
    Base class for models with added support for GradCam activation map
    and a SentiNet defense. The GradCam design is taken from:
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    If you are not planning to utilize SentiNet defense just import any model
    you like for your tasks.
    """

    def __init__(self):
        super().__init__()
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def switch_grads(self, enable=True):
        for i, n in self.named_parameters():
                n.requires_grad_(enable)

    def features(self, x):
        """
        Get latent representation, eg logit layer.
        :param x:
        :return:
        """
        raise NotImplemented

    def forward(self, x, latent=False):
        raise NotImplemented
