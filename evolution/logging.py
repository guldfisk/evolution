from __future__ import annotations

import itertools
import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from evolution.model import Generation


LogFrame = t.List[t.Any]


class LoggingOperation(ABC):
    @abstractmethod
    def inspect(self, values: LogFrame) -> t.Any:
        pass


class IndividualLoggingOperation(LoggingOperation):
    @abstractmethod
    def inspect(self, generation: Generation) -> t.Any:
        pass


class FitnessLoggingOperation(LoggingOperation):
    @abstractmethod
    def inspect(self, fitnesses: np.ndarray) -> float:
        pass


class LogAverage(FitnessLoggingOperation):
    def inspect(self, fitnesses: np.ndarray) -> float:
        return np.mean(fitnesses)


class LogMedian(FitnessLoggingOperation):
    def inspect(self, fitnesses: np.ndarray) -> float:
        return np.median(fitnesses)


class LogMax(FitnessLoggingOperation):
    def inspect(self, fitnesses: np.ndarray) -> float:
        return np.max(fitnesses)


class LogAverageConstraint(IndividualLoggingOperation):
    def __init__(self, index: int):
        self._index = index

    def inspect(self, generation: Generation) -> float:
        return np.mean([individual.fitness[self._index] for individual in generation])


class LogMaxConstraint(IndividualLoggingOperation):
    def __init__(self, index: int):
        self._index = index

    def inspect(self, generation: Generation) -> float:
        return np.max([individual.fitness[self._index] for individual in generation])


class Logger(object):
    def __init__(self, operations: t.Optional[OrderedDict[str, LoggingOperation]] = None):
        self._operations = operations if operations is not None else OrderedDict()

        self._cache_fitnesses = any(
            isinstance(operation, FitnessLoggingOperation) for operation in self._operations.values()
        )

        self._values: t.List[LogFrame] = []

    @property
    def operations(self) -> OrderedDict[str, LoggingOperation]:
        return self._operations

    @property
    def labels(self) -> t.Iterable[str]:
        return self._operations.keys()

    @property
    def values(self) -> t.Sequence[t.Sequence[t.Any]]:
        return self._values

    def _get_frame(self, generation: Generation) -> LogFrame:
        fitnesses = np.asarray([individual.fitness[0] for individual in generation]) if self._cache_fitnesses else None
        return [
            operation.inspect(fitnesses if isinstance(operation, FitnessLoggingOperation) else generation)
            for operation in self._operations.values()
        ]

    def add_frame(self, generation: Generation) -> LogFrame:
        frame = self._get_frame(generation)
        self._values.append(frame)
        return frame


def print_log_frame(n_frame: int, frame: LogFrame):
    print(
        *(
            str(arg).ljust(32)
            for arg in itertools.chain(
                (n_frame,),
                frame,
            )
        )
    )
