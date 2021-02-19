# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simultaneous perturbation stochastic approximation"""

import autograd
from pennylane.utils import _flatten, unflatten
import numpy as np


class SPSAOptimizer:
    r"""SPSA algorithm which uses small perturbations of the objective
    function in order to approximate the gradient.

    A step of the SPSA optimizer computes the new values via the rule

    .. math::

        x^{(t+1)} = x^{(t)} -  a_t g_t(x^{(t)}).

    where :math:`a_t` is a hyperparameter corresponding to step size with a
    user-defined initial value :math:`a` and :math:`g_t(\cdot)` is an
    estimate of the gradient obtained by simultaneous perturbation.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
    """

    def __init__(self,  a=0.602, c=0.101, gamma=0.1):
        self._a = a
        self._c = c
        self._gamma = gamma

        self._num_step = 1

    def update_stepsize(self, stepsize):
        r"""Update the initialized stepsize value :math:`\eta`.

        This allows for techniques such as learning rate scheduling.

        Args:
            stepsize (float): the user-defined hyperparameter :math:`\eta`
        """
        self._stepsize = stepsize

    def step(self, objective_fn, x, grad_fn=None):
        """Update x with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            x (array): NumPy array containing the current values of the variables to be updated
            grad_fn (function): Optional gradient function of the
                objective function with respect to the variables ``x``.
                If ``None``, the gradient function is computed automatically.

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """

        ak = a/(self._num_step + A) ** self._alpha
        ck = c/self._num_step ** self._gamma

        g = self.compute_grad(objective_fn, x, grad_fn=grad_fn)

        x_out = self.apply_grad(g, x)

        self._num_step += 1
        return x_out

    @staticmethod
    def estimate_grad(objective_fn, x):
        r"""Compute gradient estimate of the objective_fn at the point x.

        Args:
            objective_fn (function): the objective function for optimization
            x (array): NumPy array containing the current values of the variables to be updated

        Returns:
            array: NumPy array containing the gradient estimate :math:`g(x^{(t)})`
        """
        g = (objective_fn(x + c * delta) - objective_fn(x - c * delta)) / (2*delta)
        return g

    def apply_grad(self, grad, x):
        r"""Update the variables x to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """

        x_flat = _flatten(x)
        grad_flat = _flatten(grad)

        x_new_flat = [e - self._stepsize * g for g, e in zip(grad_flat, x_flat)]

        return unflatten(x_new_flat, x)
