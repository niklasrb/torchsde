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

import warnings

import torch

from . import base_sde
from . import methods
from . import misc
from . import sdeint
from .base_event import BaseEvent
from .._brownian import BaseBrownian, BrownianInterval
from ..settings import LEVY_AREA_APPROXIMATIONS, METHODS, NOISE_TYPES, SDE_TYPES
from ..types import Any, Dict, Optional, Scalar, Tensor, Tensors, TensorOrTensors, Vector, Sequence, Tuple


def solve_sde(sde,
           y0: Tensor,
           tspan: tuple,
           bm: Optional[BaseBrownian] = None,
           method: Optional[str] = None,
           events: Sequence[BaseEvent] = None,
           dt: Scalar = 1e-3,
           rtol: Scalar = 1e-5,
           atol: Scalar = 1e-4,
           dt_min: Scalar = 1e-3,
           adaptive = True,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None,
           logqp: bool = False,
           extra: bool = False,
           extra_solver_state: Optional[Tensors] = None,
           **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE.

    Args:
        sde: Object with methods `f` and `g` representing the
            drift and diffusion. The output of `g` should be a single tensor of
            size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m)
            for SDEs of other noise types; d is the dimensionality of state and
            m is the dimensionality of Brownian motion.
        y0 (Tensor): A tensor for the initial state.
        tspan (tuple): Integration Interval.
            The state at the first time of `tspan` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`.
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        events (sequence, optional): The list of BaseEvent events that will be
            tracked by the solver. Is eventually returned.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
        extra (bool, optional): If `True`, also return the extra hidden state
            used internally in the solver.
        extra_solver_state: (tuple of Tensors, optional): Additional state to
            initialise the solver with. Some solvers keep track of additional
            state besides y0, and this offers a way to optionally initialise
            that state.

    Returns:
        ts: A time step tensor of size (T, batch_size)
        ys: A state tensor of size (T, batch_size, d).
        events: The list of events that was passed to the solver
        if logqp is True, then the log-ratio penalty is also returned.
        If extra is True, the any extra internal state of the solver is also
        returned.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or if `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`solve_sde`")
    del unused_kwargs

    sde, y0, tspan, bm, method, events, options = check_contract(sde, y0, tspan, bm, method, events, adaptive=adaptive, options=options, names=names, logqp=logqp)
    misc.assert_no_grad(['tspan', 'dt', 'rtol', 'atol', 'dt_min'],
                        [tspan, dt, rtol, atol, dt_min])

    solver_fn = methods.select(method=method, sde_type=sde.sde_type)
    solver = solver_fn(
        sde=sde,
        bm=bm,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        extra_solver_state = solver.init_extra_solver_state(tspan[0], y0)
    ts, ys, events, extra_solver_state = solver.integrate_with_events( y0, tspan, extra_solver_state, events)

    return parse_return(ts, y0, ys, events, extra_solver_state, extra, logqp)


def check_contract(sde, y0, tspan, bm, method, events, adaptive, options, names, logqp):

    sde, y0, tspan, bm, method, options = sdeint.check_contract(sde, y0, tspan, bm, method, adaptive, options, names, logqp)

    if len(tspan) >2:
        warnings.warn("len(tspan) = {len(tspan)} > 2: Only the first and last element of tspan will be considered and time steps are returned adaptively.")

    if events is None:
        events = []
    for event in events:
        if not hasattr(event, 'step_accepted') or not hasattr(event, 'terminal'):
            raise ValueError("Invalid event: {event} missing 'step_accepted' or 'terminal' attributes.")


    return sde, y0, tspan, bm, method, events, options


def parse_return(ts, y0, ys, events, extra_solver_state, extra, logqp):
    if logqp:
        ys, log_ratio = ys.split(split_size=(y0.size(1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack(
            [log_ratio_t_plus_1 - log_ratio_t
             for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0
        ).squeeze(dim=2)

        if extra:
            return ts, ys, events, log_ratio_increments, extra_solver_state
        else:
            return ts, ys, events, log_ratio_increments
    else:
        if extra:
            return ts, ys, events, extra_solver_state
        else:
            return ts, ys, events,


