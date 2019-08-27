from __future__ import annotations

import typing as t

import functools
import operator
import random
import copy

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np


class Constraint(ABC):
    description = 'A constraint'  # type: str

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


I = t.TypeVar('I', bound = Individual)
Generation = t.List[Individual]


class LoggingOperation(ABC):

    @abstractmethod
    def inspect(self, values: t.List[t.Any]) -> t.Any:
        pass


class IndividualLoggingOperation(LoggingOperation):

    @abstractmethod
    def inspect(self, generation: Generation) -> t.Any:
        pass


class FitnessLoggingOperation(LoggingOperation):

    @abstractmethod
    def inspect(self, fitnesses: np.ndarray) -> t.Any:
        pass


class LogAverage(FitnessLoggingOperation):

    def inspect(self, fitnesses: np.ndarray) -> t.Any:
        return np.mean(fitnesses)


class LogMedian(FitnessLoggingOperation):

    def inspect(self, fitnesses: np.ndarray) -> t.Any:
        return np.median(fitnesses)


class LogMax(FitnessLoggingOperation):

    def inspect(self, fitnesses: np.ndarray) -> t.Any:
        return np.max(fitnesses)


class Logger(object):

    def __init__(self, operations: t.Optional[OrderedDict[str, LoggingOperation]] = None):
        self._operations = operations if operations is not None else OrderedDict()
        
        self._cache_fitnesses = any(
            isinstance(operation, FitnessLoggingOperation)
            for operation in
            self._operations.values()
        )

        self._values: t.List[t.List[t.Any]] = []

    @property
    def labels(self) -> t.Iterable[str]:
        return self._operations.keys()

    @property
    def values(self) -> t.Sequence[t.Sequence[t.Any]]:
        return self._values

    def add_frame(self, generation: Generation) -> None:
        fitnesses = (
            np.asarray([individual.fitness[0] for individual in generation])
            if self._cache_fitnesses else
            None
        )
        self._values.append(
            [
                operation.inspect(
                    fitnesses
                    if isinstance(operation, FitnessLoggingOperation) else
                    generation
                )
                for operation in
                self._operations.values()
            ]
        )
            

class Environment(t.Generic[I]):

    def __init__(
        self,
        individual_factory: t.Callable[[], I],
        initial_population_size: int,
        mutate: t.Callable[[I, Environment], I],
        mate: t.Callable[[I, I, Environment], t.Tuple[I, I]],
        constraints: ConstraintSet,
        logger: t.Optional[Logger] = None,
    ):
        self._individual_factory = individual_factory
        self._initial_population_size = initial_population_size
        self._generations: t.List[Generation] = []

        self._mutate = mutate
        self._mate = mate

        self._constraints: ConstraintSet = constraints

        self._logger: Logger = logger if logger is not None else Logger()

        self._mutate_threshold: float = .3
        self._mate_threshold: float = .3
        self._tournament_size: int = 4

    @classmethod
    def print(cls, *args, **kwargs):
        print(*(str(arg).ljust(35) for arg in args), **kwargs)

    def _get_initial_generation(self) -> Generation:
        generation = [
            self._individual_factory()
            for _ in
            range(self._initial_population_size)
        ]
        for individual in generation:
            setattr(individual, '_fitness', self._constraints.score(individual))

        return generation

    def mutate_population(self, generation: Generation) -> None:
        for individual in generation:
            if random.random() < self._mutate_threshold:
                self._mutate(individual, self)
                setattr(individual, '_changed', True)

    def mate_population(self, generation: Generation) -> None:
        for first, second in zip(
            generation[0::2],
            generation[1::2],
        ):
            if random.random() < self._mate_threshold:
                self._mate(first, second, self)
                setattr(first, '_changed', True)
                setattr(second, '_changed', True)

    def spawn_generation(self) -> Environment:
        if not self._generations:
            new_generation = self._get_initial_generation()

        else:
            new_generation = [
                copy.deepcopy(
                    sorted(
                        random.sample(
                            self._generations[-1],
                            self._tournament_size,
                        ),
                        key=lambda _individual: _individual.fitness[0],
                    )[-1]
                )
                for _ in
                range(self._initial_population_size)
            ]

            for individual in new_generation:
                if hasattr(individual, '_changed'):
                    delattr(individual, '_changed')

            self.mate_population(new_generation)
            self.mutate_population(new_generation)

            for individual in new_generation:
                if hasattr(individual, '_changed'):
                    setattr(individual, '_fitness', self._constraints.score(individual))

        self._generations.append(
            new_generation
        )

        self._logger.add_frame(new_generation)
        self.print(
            len(self._generations) - 1,
            *self._logger.values[-1],
        )

        return self
    
    def spawn_generations(self, amount: int) -> Environment:
        for _ in range(amount):
            self.spawn_generation()
        return self
    
    def fittest(self) -> I:
        return sorted(
            self._generations[-1],
            key = lambda individual: individual.fitness,
        )[-1]
