
import abc

import torch
from torch import nn
import time

from . import misc
from ..settings import NOISE_TYPES, SDE_TYPES
from ..types import Tensor

class BaseEvent(abc.ABC, nn.Module):
    """
    Base class for SDE events

    Attributes:
    terminal (Boolean): Whether this event can cause the integration to terminate for a given batch
    """

    def __init__(self, terminal = False):
        super(BaseEvent, self).__init__()
        self.terminal = terminal


    @abc.abstractmethod
    def step_accepted(self, solver, evolved_batches, t, y, t_last, y_last):
        """
        This placeholder function is called every time a solver step has been accepted

        Attributes:
            solver: The SDESolver instance
            evolved_batches: The list of currently evolved batches
            t: accepted step time
            y: accepted step state
            t_last: last accepted step time
            y_last: last accepted step state

        Returns:
            None, or List[int] the list of batches to terminate if self.terminal == True
        """
        raise NotImplementedError


class VerboseEvent(BaseEvent):
    """
    Event that outputs information about the integration process at given intervals/steps

    Attributes:
    n_steps (int) : Output at every nth step - 0 for no step counting
    """

    def __init__(self, n_steps=1, avg=False):
        super(VerboseEvent, self).__init__(terminal=False)
        self.n_steps = n_steps
        self._step_counter = 0
        self._avg = avg and n_steps > 1
        self._tic = 0
        self._ra_t = 0 # running average t
        self._ra_ee = 0 # running average error estime

    def output_info(self, solver, evolved_batches, t, y, t_last, y_last):
        """
        This function is called whenever the condition for output is met
        """
        if self._avg:
            print(f"step={solver.i}, t={t}, step_size={solver.dt}, avg(error_estimate)={self._ra_ee/self.n_steps}, avg(real time)={self._ra_t/self.n_steps}, y={y}, evolved_batches={evolved_batches}")
        else:
            print(f"step={solver.i}, t={t}, step_size={solver.dt}, error_estimate={solver.error_estimate}, y={y}, evolved_batches={evolved_batches}")


    def step_accepted(self, solver, evolved_batches, t, y, t_last, y_last):
        """
        Checks if the condition for output is met at every accepted step
        """
        toc = time.perf_counter()
        self._step_counter += 1
        if self._avg:
            self._ra_ee += solver.error_estimate
            self._ra_t += toc - self._tic if self._tic > 0 else 0.
        if self.n_steps > 0 and self._step_counter >= self.n_steps:
            self.output_info(solver, evolved_batches, t, y, t_last, y_last)
            self._step_counter = 0
            if self._avg:
                self._ra_ee = 0.
                self._ra_t = 0.
        self._tic = toc


class HitTargetEvent(BaseEvent):
    """
    An event that searches for roots in the self.target function. On hit, the interpolated time is saved in self.triggered as (t, batch)

    For self.unique = True, only the first instance for each batch is saved
    """

    def __init__(self, terminal = False, unique = False):
        super(HitTargetEvent, self).__init__(terminal = terminal)
        self.triggered = []
        self.unique = unique


    def find_root(t_0, y_0, t_1, y_1):
         """
         Linearly interpolates the hitting time
         """
         return t_0 - (t_1 - t_0)/(y_1 - y_0) * y_0


    @abc.abstractmethod
    def target(self, t, y):
        """
        The target function. This triggers the event each time a root is found for a given batch

        Returns
            Tensor of size (batch_size)
        """
        raise NotImplementedError


    def step_accepted(self, solver, evolved_batches, t, y, t_last, y_last):
        """
        Is called every time a step is accepted and searches for a root in self.target in between the current and last step
        """
        h = self.target(t,y)
        h_last = self.target(t_last, y_last)
        batch_roots = torch.where(h * h_last < 0)[0]

        if len(batch_roots) > 0:
            batches = []
            for i in batch_roots:
                batch = evolved_batches[i]
                t_root = HitTargetEvent.find_root(t, h[i], t_last, h_last[i])

                if not self.unique or not batch in [b for _,b in self.triggered]:
                    self.triggered.append((t_root, batch))
                batches.append(batch)

            if self.terminal:
                return batches


