from __future__ import annotations

import typing as t

from abc import ABC


class Individual(ABC):

    @property
    def weighted_fitness(self) -> float:
        return getattr(self, '_fitness')[0]

    @property
    def fitness(self) -> t.Tuple[float, ...]:
        return getattr(self, '_fitness')


Generation = t.List[Individual]
