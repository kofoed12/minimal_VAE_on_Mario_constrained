#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from __future__ import annotations

from abc import ABC
from typing import Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""
    Base class for analytic acquisition functions.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        posterior_transform = self._deprecate_acqf_objective(
            posterior_transform=posterior_transform,
            objective=kwargs.get("objective"),
        )
        if posterior_transform is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a "
                    "multi-output model."
                )
        else:
            if not isinstance(posterior_transform, PosteriorTransform):
                raise UnsupportedError(
                    "AnalyticAcquisitionFunctions only support PosteriorTransforms."
                )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )


class ExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei
