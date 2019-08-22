from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy

import functools
import operator
import random

import numpy as np


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

        self._frames: t.List[OrderedDict[str, t.Any]] = []

    def get_log_frame(self, generation: Generation) -> OrderedDict[str, t.Any]:
        fitnesses = (
            np.asarray((individual.fitness for individual in generation))
            if self._cache_fitnesses else
            None
        )
        frame = OrderedDict(
            [
                (
                    key,
                    operation.inspect(
                        fitnesses
                        if isinstance(operation, FitnessLoggingOperation) else
                        generation
                    ),
                )
                for key, operation in
                self._operations.items()
            ]
        )

        self._frames.append(frame)
        return frame


class Environment(ABC):

    def __init__(
        self,
        individual_factory: t.Callable[[], Individual],
        initial_population_size: int,
        mutate: t.Callable[[Individual], Individual],
        mate: t.Callable[[Individual, Individual], t.Tuple[Individual, Individual]],
        logger: t.Optional[Logger] = None,
    ):
        self._individual_factory = individual_factory
        self._initial_population_size = initial_population_size
        self._generations: t.List[Generation] = [
            [
                individual_factory()
                for _ in
                range(initial_population_size)
            ]
        ]

        self._mutate = mutate
        self._mate = mate

        self._logger = logger if logger is not None else Logger()

        self._mutate_threshold = .25
        self._mate_threshold = .2
        self._tournament_size = 4

    @classmethod
    def print(cls, *args, **kwargs):
        print(*(str(arg).ljust(50) for arg in args), **kwargs)

    def mutate_population(self) -> None:
        for individual in self._generations[-1]:
            if random.random() < self._mutate_threshold:
                self._mutate(individual)

    def mate_population(self) -> None:
        for first, second in zip(
            self._generations[-1][0::2],
            self._generations[-1][1::2],
        ):
            if random.random() < self._mate_threshold:
                self._mate(first, second)

    def spawn_generation(self) -> Environment:
        new_generation = [
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

        for individual in new_generation:
            delattr(individual, '_fitness')

        self._generations.append(
            new_generation
        )
        
        self.mate_population()
        self.mutate_population()

        frame = self._logger.get_log_frame(new_generation)
        self.print(frame.values())

        return self
    
    def spawn_generations(self, amount: int) -> Environment:
        for _ in range(amount):
            self.spawn_generation()
        return self
    
    def fittest(self) -> Individual:
        return sorted(
            self._generations[-1],
            key = lambda individual: individual.fitness,
        )[-1]
