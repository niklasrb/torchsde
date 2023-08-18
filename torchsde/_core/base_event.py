
import abc

import torch
from torch import nn

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

            