# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import warnings

import torch

from . import adaptive_stepping
from . import better_abc
from . import interp
from . import misc
from .base_event import BaseEvent
from .base_sde import BaseSDE
from .._brownian import BaseBrownian, BrownianInterval
from ..settings import NOISE_TYPES
from ..types import Scalar, Tensor, Dict, Tensors, Tuple, Sequence


class BaseSDESolver(metaclass=better_abc.ABCMeta):
    """API for solvers with possibly adaptive time stepping."""

    strong_order = better_abc.abstract_attribute()
    weak_order = better_abc.abstract_attribute()
    sde_type = better_abc.abstract_attribute()
    noise_types = better_abc.abstract_attribute()
    levy_area_approximations = better_abc.abstract_attribute()

    def __init__(self,
                 sde: BaseSDE,
                 bm: BaseBrownian,
                 dt: Scalar,
                 adaptive: bool,
                 rtol: Scalar,
                 atol: Scalar,
                 dt_min: Scalar,
                 options: Dict,
                 **kwargs):
        super(BaseSDESolver, self).__init__(**kwargs)
        if sde.sde_type != self.sde_type:
            raise ValueError(f"SDE is of type {sde.sde_type} but solver is for type {self.sde_type}")
        if sde.noise_type not in self.noise_types:
            raise ValueError(f"SDE has noise type {sde.noise_type} but solver only supports noise types "
                             f"{self.noise_types}")
        if bm.levy_area_approximation not in self.levy_area_approximations:
            raise ValueError(f"SDE solver requires one of {self.levy_area_approximations} set as the "
                             f"`levy_area_approximation` on the Brownian motion.")
        if sde.noise_type == NOISE_TYPES.scalar and torch.Size(bm.shape[1:]).numel() != 1:  # noqa
            raise ValueError("The Brownian motion for scalar SDEs must of dimension 1.")

        self.sde = sde
        self.bm = bm
        self.dt = dt
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.options = options

    def __repr__(self):
        return f"{self.__class__.__name__} of strong order: {self.strong_order}, and weak order: {self.weak_order}"

    def init_extra_solver_state(self, t0, y0) -> Tensors:
        return ()

    @abc.abstractmethod
    def step(self, t0: Scalar, t1: Scalar, y0: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Propose a step with step size from time t to time next_t, with
         current state y.

        Args:
            t0: float or Tensor of size (,).
            t1: float or Tensor of size (,).
            y0: Tensor of size (batch_size, d).
            extra0: Any extra state for the solver.

        Returns:
            y1, where y1 is a Tensor of size (batch_size, d).
            extra1: Modified extra state for the solver.
        """
        raise NotImplementedError

    def integrate(self, y0: Tensor, ts: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        step_size = self.dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]
        prev_error_ratio = None

        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if self.adaptive:
                    # Take 1 full step.
                    next_y_full, _ = self.step(curr_t, next_t, curr_y, curr_extra)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y, midpoint_extra = self.step(curr_t, midpoint_t, curr_y, curr_extra)
                    next_y, next_extra = self.step(midpoint_t, next_t, midpoint_y, midpoint_extra)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, self.rtol, self.atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < self.dt_min:
                        warnings.warn("Hitting minimum allowed step size in adaptive time-stepping.")
                        step_size = self.dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= self.dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y, curr_extra = next_t, next_y, next_extra
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                    curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra


    def integrate_with_events(self, y0: Tensor, tspan: Tensor, extra0: Tensors, events : Sequence[BaseEvent]) -> Tuple[Tensor, Tensor, Sequence[BaseEvent], Tensors]:
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ts, where ts is a Tensor of size (T, batch_size)
            ys, where ys is a Tensor of size (T, batch_size, d).
            events, where the considered events are returned
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        batch_size = y0.shape[0]
        evolved_batches = torch.arange(batch_size)

        prev_t = curr_t = tspan[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]
        ts = [ torch.Tensor([tspan[0]]*batch_size) ]
        prev_error_ratio = None
        self.i = 0

        status = 0
        while status == 0:

            accept_step = False
            while not accept_step:
                next_t = min(curr_t + self.dt, tspan[-1])
                # Take 1 full step.
                next_y_full, _ = self.step(curr_t, next_t, curr_y, curr_extra)
                # Take 2 half steps.
                midpoint_t = 0.5 * (curr_t + next_t)
                midpoint_y, midpoint_extra = self.step(curr_t, midpoint_t, curr_y, curr_extra)
                next_y, next_extra = self.step(midpoint_t, next_t, midpoint_y, midpoint_extra)

                if misc.is_nan(next_y_full) or misc.is_nan(next_y):
                    if self.dt == self.dt_min or not self.adaptive:
                        warnings.warn("Found nans in the integration with minimum stepsize. Terminating Integration")
                        status = -1
                        break
                    warnings.warn("Found nans in the integration.")
                    self.dt = min(self.dt/2., self.dt_min)
                    continue

                # Estimate error based on difference between 1 full step and 2 half steps.
                with torch.no_grad():
                    self.error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, self.rtol, self.atol)
                    if self.adaptive:
                            self.dt, prev_error_ratio = adaptive_stepping.update_step_size(
                                error_estimate=self.error_estimate,
                                prev_step_size=self.dt,
                                prev_error_ratio=prev_error_ratio
                            )
                    else:
                        if self.error_estimate > 1.:
                            warnings.warn("Large errors with given stepsize. Consider lowering stepsize or making it adaptive.")

                if self.dt < self.dt_min:
                    warnings.warn("Hitting minimum allowed step size in adaptive time-stepping.")
                    self.dt = self.dt_min
                    prev_error_ratio = None

                # Accept step.
                if self.error_estimate <= 1 or self.dt <= self.dt_min or not self.adaptive:
                    prev_t, prev_y = curr_t, curr_y
                    curr_t, curr_y, curr_extra = next_t, next_y, next_extra
                    accept_step = True

            if status == -1:
                break
            # Only add batches that changed, the others stay constant
            ts.append(ts[-1].detach().clone())
            ys.append(ys[-1].detach().clone())
            ts[-1][evolved_batches] = curr_t
            ys[-1][evolved_batches] = curr_y
            self.i += 1

            # Handle Events
            terminate_batches = []
            for event  in events:
                if not event.terminal:
                    event.step_accepted(self, evolved_batches, curr_t, curr_y, prev_t, prev_y)
                    continue

                terminate = event.step_accepted(self, evolved_batches, curr_t, curr_y, prev_t, prev_y)

                if terminate is not None and len(terminate) > 0:
                    for tb in terminate:
                        terminate_batches.append(tb)

            # Reduce active batch size if needed
            if len(terminate_batches) > 0:
                terminate_batches = list(set(terminate_batches))
                for tb in terminate_batches:
                    evolved_batches = evolved_batches[evolved_batches != tb]

                new_batch_size = len(evolved_batches)
                if new_batch_size == 0:
                    status = 2
                    continue

                # reduce brownian motion size
                bm_size = self.bm._size
                new_bm_size = (new_batch_size, *bm_size[1:])
                self.bm = BrownianInterval( t0 = self.bm._start,
                                            t1 = self.bm._end,
                                            size= new_bm_size,
                                            dtype = self.bm._dtype,
                                            device = self.bm._device,
                                            entropy = self.bm._entropy,
                                            dt = self.bm._dt,
                                            tol = self.bm._tol,
                                            pool_size = self.bm._pool_size,
                                            cache_size = self.bm._cache_size,
                                            halfway_tree = self.bm._halfway_tree,
                                            levy_area_approximation = self.bm._levy_area_approximation)


            curr_y = ys[-1][evolved_batches]

            if status == 0 and curr_t >= tspan[-1]:
                status = 1


        return torch.stack(ts, dim=0), torch.stack(ys, dim=0), events,  curr_extra
