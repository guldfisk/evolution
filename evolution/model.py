from __future__ import annotations

import typing as t

import functools
import operator

from abc import ABC, abstractmethod


class Constraint(ABC):
    description: str = 'A constraint'

    @abstractmethod
    def score(self, individual: Individual) -> float:
        pass


class ConstraintSet(object):

    def __init__(self, constraints: t.Iterable[t.Tuple[Constraint, float]]):
        self._constraints = tuple(constraint for constraint, _ in constraints)
        self._weights = tuple(weight for _, weight in constraints)

    def score(self, individual: Individual) -> t.Tuple[float, ...]:
        unweighted_values = tuple(
            constraint.score(individual)
            for constraint in
            self._constraints
        )

        return (
           functools.reduce(
               operator.mul,
               (
                   value ** weight
                   for value, weight in
                   zip(unweighted_values, self._weights)
               )
           ),
       ) + unweighted_values

    def total_score(self, individual: Individual) -> float:
        return self.score(individual)[0]

    def __iter__(self) -> t.Iterable[Constraint]:
        return self._constraints.__iter__()


class Individual(ABC):

    @property
    def weighted_fitness(self) -> float:
        return getattr(self, '_fitness')[0]

    @property
    def fitness(self) -> t.Tuple[float, ...]:
        return getattr(self, '_fitness')

    def score(self, constraints: ConstraintSet) -> t.Tuple[float, ...]:
        return constraints.score(self)


Generation = t.List[Individual]
