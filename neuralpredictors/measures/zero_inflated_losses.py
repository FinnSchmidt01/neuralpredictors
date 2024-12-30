import torch
import torch.nn as nn
from torch.distributions import (
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
    StudentT,
)


class ZeroInflationLossBase(nn.Module):
    def __init__(self, avg=False, per_neuron=False, return_logdet=False):
        super().__init__()
        self.avg = avg
        self.per_neuron = per_neuron
        self.return_logdet = return_logdet

    def forward(self, target, output, **kwargs):
        *slab_params, loc, q = output
        neurons_n = target.shape[1]

        if loc.requires_grad:
            self.multi_clamp(loc, [0.0] * neurons_n, target.max(dim=0)[0])

        zero_mask = target <= loc
        nonzero_mask = target > loc

        # spike loss
        spike_logl = torch.log(1 - q) - torch.log(loc)
        spike_logl = spike_logl * zero_mask
        spike_logl = spike_logl.sum(dim=0)

        if self.avg:
            denominator = zero_mask.sum(dim=0)
            denominator = torch.where(
                (denominator != 0.0) & (spike_logl != 0.0), denominator, 1.0
            )  # avoid division by 0
            spike_logl = spike_logl / denominator

        if not self.per_neuron:
            spike_logl = spike_logl.mean() if self.avg else spike_logl.sum()

        # make sure loc is always smaller than the smallest non-zero response
        if loc.requires_grad:
            self.multi_clamp(
                loc,
                [0.0] * neurons_n,
                self.find_nonzero_min(target * nonzero_mask) * 0.999,
            )

        # slab loss
        slab_logl, logdet = self.get_slab_logl(
            *slab_params, loc=loc, q=q, target=target, zero_mask=zero_mask, nonzero_mask=nonzero_mask, **kwargs
        )

        slab_logl = slab_logl * nonzero_mask

        slab_logl = slab_logl.sum(dim=0)
        if self.avg:
            denominator = nonzero_mask.sum(dim=0)
            denominator = torch.where(
                (denominator != 0.0) & (slab_logl != 0.0), denominator, 1.0
            )  # avoid division by 0
            slab_logl = slab_logl / denominator

        if not self.per_neuron:
            slab_logl = slab_logl.mean() if self.avg else slab_logl.sum()

        # total loss
        loss = -(spike_logl + slab_logl)

        assert not (torch.isinf(loss).any() or torch.isnan(loss).any()), "Loss is nan or inf"
        if self.return_logdet:
            return loss, logdet
        else:
            return loss

    def get_slab_logl(self, *slab_params, loc, q, target, zero_mask, nonzero_mask, **kwargs):
        """
        Compute the logl of the slab part of the zero-inflated distribution. Must return both the logl and the
        log-determinant of the transformed slab responses. If the slab responses were not transformed, return logdet=0
        """
        raise NotImplementedError()

    @staticmethod
    def multi_clamp(tensor, mins, maxs):
        for tensor_row in tensor:
            for tensorval, minval, maxval in zip(tensor_row, mins, maxs):
                tensorval.data.clamp_(minval, maxval)

    @staticmethod
    def find_nonzero_min(tensor, dim=0):
        tensor[tensor == 0.0] += tensor.max()
        return tensor.min(dim)[0]


class ZIGLoss(ZeroInflationLossBase):
    def get_slab_logl(self, *slab_params, loc, q, target, zero_mask, nonzero_mask, **kwargs):
        theta, k = slab_params
        logdet = 0
        slab_logl = (
            torch.log((1 - q) * zero_mask + 1 * nonzero_mask)  # - torch.log(loc*zero_mask+1*nonzero_mask)
            + torch.log(q * nonzero_mask + 1.0 * zero_mask)
            + (k - 1) * torch.log(target * nonzero_mask - loc * nonzero_mask + 1.0 * zero_mask)
            - (target * nonzero_mask - loc * nonzero_mask) / theta
            - k * torch.log(theta)
            - torch.lgamma(k)
        )
        return slab_logl, logdet


class ZILLoss(ZeroInflationLossBase):
    def get_slab_logl(self, *slab_params, loc, q, target, zero_mask, nonzero_mask, **kwargs):
        mean, variance = slab_params
        logdet = -torch.log((target - loc) * nonzero_mask + 1.0 * zero_mask)
        slab_logl = (
            torch.log(q)
            + Normal(mean, torch.sqrt(variance)).log_prob(torch.log((target - loc) * nonzero_mask + 1.0 * zero_mask))
            + logdet
        )
        return slab_logl, logdet


class ZILogStudentTLoss(ZeroInflationLossBase):
    def get_slab_logl(self, *slab_params, loc, q, target, zero_mask, nonzero_mask, **kwargs):
        df, location, scale = slab_params
        logdet = -torch.log((target - loc) * nonzero_mask + 1.0 * zero_mask)
        slab_logl = (
            torch.log(q)
            + StudentT(df=df, loc=location, scale=scale).log_prob(
                torch.log((target - loc) * nonzero_mask + 1.0 * zero_mask)
            )
            + logdet
        )
        return slab_logl, logdet


class ZIFLoss(ZeroInflationLossBase):
    def get_slab_logl(self, model, data_key, targets, rho, qs, means, psi_diag):
        psi_diag = psi_diag.to(means.device)
        # Create masks for spike and slab
        uniform_masks = (targets < rho).detach()  # spike
        gaussian_masks = (targets >= rho).detach()  # slab

        # Compute spike log-likelihood
        uniform_log_probs = torch.log(1 - qs)  # - torch.log(rho)  # Shape: (Batch, Time, Neurons, Samples)
        uniform_log_probs = uniform_log_probs * uniform_masks  # Shape: (Batch, Time, Neurons, Samples)

        # Compute slab log-likelihood with flow transformation
        transformed_targets, logdets = model.flow[data_key](
            targets.squeeze(-1), zero_mask=uniform_masks.squeeze(-1).int()
        )  # Shape: (Batch, Time, Neurons)
        # Extract the dimensions
        (
            batch_size,
            time_steps,
            neurons,
        ) = transformed_targets.shape

        # Compute the covariance matrix
        # Only using diagonal psi_diag, so the covariance is diagonal
        psi_inv = 1.0 / psi_diag  # Shape: (Neurons,)

        # Compute the difference between targets and means
        diff = transformed_targets.unsqueeze(-1) - means  # Shape: (Batch, Time, Neurons, Samples)
        # Compute the quadratic term in the Gaussian log-likelihood
        # term = diff^T * inv_covariance_matrix * diff, computed for each element
        quadratic_term = torch.einsum("btis,i,btis->btis", diff, psi_inv, diff)

        # Compute log determinant of the diagonal covariance matrix
        logdet_covariance = torch.log(psi_diag).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Compute the log-likelihood for the slab (Gaussian) part
        gaussian_log_probs = -0.5 * (
            quadratic_term + torch.log(torch.tensor(2 * torch.pi)) + logdet_covariance
        )  # Shape: (Batch, Time,, Neurons Samples)
        gaussian_log_probs = (
            gaussian_log_probs + logdets.unsqueeze(-1) + torch.log(qs)
        )  # Shape: (Batch, Time,, Neurons Samples)
        gaussian_log_probs = gaussian_log_probs * gaussian_masks.int()
        # Combine spike and slab log-likelihoods
        loglikelihood = uniform_log_probs + gaussian_log_probs  # Scalar

        return loglikelihood, logdets.squeeze()
