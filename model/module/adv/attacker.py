import torch
import torch.nn as nn
# from pudb import set_trace
from model.module.builder import Modules
from torch.nn.functional import cross_entropy

from utils import switch_adv


@Modules.register_module('LinfPGD')
class LinfPGD(nn.Module):
    """Projected Gradient Decent(PGD) attack."""

    def __init__(self,
                 model,
                 eps=8,
                 step_size=2,
                 num_steps=7,
                 criterion=None,
                 random_start=True,
                 targeted=False,
                 clip_min=0.,
                 clip_max=1.,
                 **kwargs):

        super(LinfPGD, self).__init__()

        self.device = next(model.parameters()).device
        self.model = model
        self.eps = torch.tensor(eps).to(self.device) / 255.
        self.step_size = torch.tensor(step_size).to(self.device) / 255
        self.num_steps = num_steps
        self.random_start = random_start
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = cross_entropy

    def attack(self, imgs, targets):
        self.training = self.model.training

        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        self.model.eval()
        self._model_freeze()

        perturbation = torch.zeros_like(imgs).to(self.device)
        if self.random_start:
            perturbation = torch.rand_like(imgs).to(device=self.device)
            perturbation = self.compute_perturbation(imgs + perturbation, imgs,
                                                     targets)

        with torch.enable_grad():
            self.model.apply(switch_adv)
            for i in range(self.num_steps):
                perturbation = self.onestep(imgs, perturbation, targets)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return imgs + perturbation

    def onestep(self, imgs, perturbation, targets):
        # Running one step for
        adv_imgs = imgs + perturbation
        adv_imgs.requires_grad = True

        self.model.apply(switch_adv)
        attack_loss = self.criterion(self.model(adv_imgs), targets)

        self.model.zero_grad()
        attack_loss.backward()
        grad = adv_imgs.grad

        # Essential: delete the computation graph to save GPU RAM
        adv_imgs.requires_grad = False

        if self.targeted:
            adv_imgs = adv_imgs.detach() - self.step_size * torch.sign(grad)
        else:
            adv_imgs = adv_imgs.detach() + self.step_size * torch.sign(grad)

        perturbation = self.compute_perturbation(adv_imgs, imgs, targets)

        return perturbation

    def compute_perturbation(self, adv_x, x, targets):
        # Project the perturbation to Lp ball
        perturbation = torch.clamp(adv_x - x, -self.eps, self.eps)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x + perturbation, self.clip_min,
                                   self.clip_max) - x

        return perturbation

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


@Modules.register_module('AdaptLinfPGD')
class AdaptLinfPGD(LinfPGD):
    """最大类->最小类：step_size, eps 按类别逐渐增大

    Set different step_size and epsilon for different categories according to
    the number of samples

    Args:
        eps: control the bound.
        step_size(1-d list): update size of each step
        num_steps: number of steps
    """

    def onestep(self, imgs, perturbation, targets):
        # Running one step for
        adv_imgs = imgs + perturbation
        adv_imgs.requires_grad = True

        self.model.apply(switch_adv)
        attack_loss = self.criterion(self.model(adv_imgs), targets)

        self.model.zero_grad()
        attack_loss.backward()
        grad = adv_imgs.grad

        # Essential: delete the computation graph to save GPU RAM
        adv_imgs.requires_grad = False

        batch_stepsize = self.step_size[targets].view(imgs.shape[0], 1, 1, 1)
        if self.targeted:
            adv_imgs = adv_imgs.detach() - batch_stepsize * torch.sign(grad)
        else:
            adv_imgs = adv_imgs.detach() + batch_stepsize * torch.sign(grad)

        perturbation = self.compute_perturbation(adv_imgs, imgs, targets)

        return perturbation

    def compute_perturbation(self, adv_x, x, targets):
        batch_eps = self.eps[targets].view(targets.shape[0], 1, 1, 1)
        perturbation = torch.min(adv_x - x, batch_eps)
        perturbation = torch.max(perturbation, -batch_eps)

        return perturbation


@Modules.register_module('AdaptLinfPGD2')
class AdaptLinfPGD2(AdaptLinfPGD):
    """最大类->最小类：8/255->16/255"""

    def project(self, perturbation, use_target, target):
        if use_target:
            batch_size = target.shape[0]
            adapt_ratio = target / 19
            epsilon = self.eps * (1 + adapt_ratio)
            epsilon = epsilon.view(batch_size, 1, 1, 1)
            perturbation_min = torch.min(perturbation, epsilon)
            ret_eps = torch.max(perturbation_min, -epsilon)
        else:
            epsilon = self.eps
            ret_eps = torch.clamp(perturbation, -epsilon, epsilon)

        return ret_eps


@Modules.register_module('AdaptLinfPGD3')
class AdaptLinfPGD3(AdaptLinfPGD):
    """最大类->最小类：4/255->16/255"""

    def project(self, perturbation, use_target, target):
        if use_target:
            batch_size = target.shape[0]
            adapt_ratio = target / 19
            epsilon = (4 + 12 * adapt_ratio) / 255
            epsilon = epsilon.view(batch_size, 1, 1, 1)
            perturbation_min = torch.min(perturbation, epsilon)
            ret_eps = torch.max(perturbation_min, -epsilon)
        else:
            epsilon = self.eps
            ret_eps = torch.clamp(perturbation, -epsilon, epsilon)

        return ret_eps


@Modules.register_module('AdaptLinfPGD4')
class AdaptLinfPGD4(AdaptLinfPGD):
    """最大类->最小类：8/255->24/255"""

    def project(self, perturbation, use_target, target):
        if use_target:
            batch_size = target.shape[0]
            adapt_ratio = target / 19
            epsilon = (8 + 16 * adapt_ratio) / 255
            epsilon = epsilon.view(batch_size, 1, 1, 1)
            perturbation_min = torch.min(perturbation, epsilon)
            ret_eps = torch.max(perturbation_min, -epsilon)
        else:
            epsilon = self.eps
            ret_eps = torch.clamp(perturbation, -epsilon, epsilon)

        return ret_eps


@Modules.register_module('AdaptLinfPGD5')
class AdaptLinfPGD5(AdaptLinfPGD):
    """最大类->最小类：4/255->24/255"""

    def project(self, perturbation, use_target, target):
        if use_target:
            batch_size = target.shape[0]
            adapt_ratio = target / 19
            epsilon = (4 + 20 * adapt_ratio) / 255
            epsilon = epsilon.view(batch_size, 1, 1, 1)
            perturbation_min = torch.min(perturbation, epsilon)
            ret_eps = torch.max(perturbation_min, -epsilon)
        else:
            epsilon = self.eps
            ret_eps = torch.clamp(perturbation, -epsilon, epsilon)

        return ret_eps


@Modules.register_module('L2PGD')
class L2PGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """

    def __init__(self,
                 model,
                 epsilon=5,
                 step=1,
                 iterations=20,
                 criterion=None,
                 random_start=True,
                 targeted=False,
                 clip_min=0.,
                 clip_max=1.,
                 **kwargs):
        super(L2PGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target:\
                nn.functional.cross_entropy(model(input), target)

        # Model status
        self.training = self.model.training

    def project(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        return perturbation.renorm(p=2, dim=0, maxnorm=self.epsilon)

    def compute_perturbation(self, adv_x, x):
        # Project the perturbation to Lp ball
        perturbation = self.project(adv_x - x)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x + perturbation, self.clip_min,
                                   self.clip_max) - x

        return perturbation

    def onestep(self, x, perturbation, target):
        # Running one step for
        adv_x = x + perturbation
        adv_x.requires_grad = True

        atk_loss = self.criterion(self.model, adv_x, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        g_norm = torch.norm(grad.view(x.shape[0], -1), p=2,
                            dim=1).view(-1, *([1] * (len(x.shape) - 1)))
        grad = grad / (g_norm + 1e-10)
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:  # decrease loss
            adv_x = adv_x.detach() - self.step * grad
        else:  # increase loss
            adv_x = adv_x.detach() + self.step * grad
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def random_perturbation(self, x):
        perturbation = torch.rand_like(x).to(device=self.device)
        perturbation = self.compute_perturbation(x + perturbation, x)

        return perturbation

    def attack(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        self.training = self.model.training
        self.model.eval()
        self._model_freeze()

        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)

        with torch.enable_grad():
            for i in range(self.iterations):
                perturbation = self.onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation
