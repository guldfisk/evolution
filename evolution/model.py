from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from copy import copy

import functools
import operator
import random

import numpy as np

# class Constraint(ABC):
#     description = 'A constraint' #type: str
# 
#     def __init__(
#         self,
#     ):
#         pass
# 
#     @abstractmethod
#     def score(self, individual: 'Individual') -> float:
#         pass
# 
# 
# class ConstraintSet(object):
# 
#     def __init__(self, constraints: t.Tuple[t.Tuple[Constraint, float], ...]):
#         self._constraints = constraints
#         self._weights = tuple(weight for _, weight in self._constraints)
# 
#     def score(self, individual: 'Individual') -> t.Tuple[float, ...]:
#         unweighted_values = tuple(
#             constraint.score(individual)
#             for constraint, _ in
#             self._constraints
#         )
# 
#         return (
#                    functools.reduce(
#                        operator.mul,
#                        (
#                            value ** weight
#                            for value, weight in
#                            zip(unweighted_values, self._weights)
#                        )
#                    ),
#                ) + unweighted_values
# 
#     def total_score(self, individual: 'Individual') -> float:
#         return self.score(individual)[0]
# 
#     def __iter__(self) -> t.Iterable[Constraint]:
#         return (constraint for constraint, _ in self._constraints)
# 
# 
# class ConstraintSetBluePrint(object):
# 
#     def __init__(self, *constraints: t.Tuple[t.Type[Constraint], float, t.Dict]):
#         self._constraints = constraints
# 
#     def realise(
#         self,
#         # constrained_nodes: t.FrozenSet[ConstrainedNode],
#         # trap_amount: int,
#         # random_population: t.Collection[TrapDistribution],
#     ) -> ConstraintSet:
#         pass
#         # return ConstraintSet(
#         #     tuple(
#         #         (
#         #             constraint_type(constrained_nodes, trap_amount, random_population, **kwargs),
#         #             weight,
#         #         )
#         #         for constraint_type, weight, kwargs in
#         #         self._constraints
#         #     )
#         # )


class Individual(ABC):

    @property
    def fitness(self) -> float:
        if not hasattr(self, '_fitness'):
            setattr(self, '_fitness', self.calc_fitness())
        return getattr(self, '_fitness')

    @abstractmethod
    def calc_fitness(self) -> float:
        pass

    @abstractmethod
    def __copy__(self) -> Individual:
        pass


# class Population(ABC):
# 
#     def __init__(self, individuals: t.List[Individual]):
#         self._individuals = individuals
# 
#     # @abstractmethod
#     # def serialize(self) -> t.AnyStr:
#     #     pass
#     #
#     # @abstractmethod
#     # def deserialize(self, s: t.AnyStr) -> 'Population':
#     #     pass


class Environment(ABC):

    def __init__(self, individual_factory: t.Callable[[], Individual], initial_population_size: int):
        self._individual_factory = individual_factory
        self._initial_population_size = initial_population_size
        self._generations = [
            [
                individual_factory()
                for _ in
                range(initial_population_size)
            ]
        ]

        self._mutate_threshold = .25
        self._mate_threshold = .2
        self._tournament_size = 4

    @abstractmethod
    def mutate(self, individual) -> Individual:
        pass

    @abstractmethod
    def mate(self, first_individual, second_individual) -> t.Tuple[Individual, Individual]:
        pass

    @classmethod
    def print(cls, *args, **kwargs):
        print(*args, sep='\t\t', **kwargs)

    def mutate_population(self) -> None:
        for individual in self._generations[-1]:
            if random.random() < self._mutate_threshold:
                self.mutate(individual)

    def mate_population(self) -> None:
        for first, second in zip(
            self._generations[-1][0::2],
            self._generations[-1][1::2],
        ):
            if random.random() < self._mate_threshold:
                self.mate(first, second)

    def next_generation(self) -> Environment:
        self._generations.append(
            [
                copy(
                    sorted(
                        random.sample(
                            self._generations[-1],
                            self._tournament_size,
                        ),
                        key=lambda individual: individual.fitness,
                    )[-1]
                )
                for _ in
                range(self._initial_population_size)
            ]
        )
        
        self.mate_population()
        self.mate_population()

        self.print(len(self._generations) - 1, self.mean(), self.fittest().fitness)

        return self
    
    def generations(self, amount: int) -> Environment:
        for _ in range(amount):
            self.next_generation()
        return self
    
    def fittest(self) -> Individual:
        return sorted(
            self._generations[-1],
            key = lambda individual: individual.fitness,
        )[-1]

    def mean(self) -> float:
        return np.mean(
            [
                individual.fitness
                for individual in
                self._generations[-1]
            ]
        )