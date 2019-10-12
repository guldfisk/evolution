from __future__ import annotations

import typing as t

import random
import copy

from evolution.logging import Logger, LogFrame, print_log_frame
from evolution.model import Individual, ConstraintSet, Generation


I = t.TypeVar('I', bound = Individual)


class Environment(t.Generic[I]):

    def __init__(
        self,
        individual_factory: t.Callable[[], I],
        initial_population_size: int,
        mutate: t.Callable[[I, Environment], I],
        mate: t.Callable[[I, I, Environment], t.Tuple[I, I]],
        constraints: ConstraintSet,
        logger: t.Optional[Logger] = None,
        save_generations: bool = True,
        *,
        print_log_frames: bool = False,
    ):
        self._individual_factory = individual_factory
        self._initial_population_size = initial_population_size
        self._generations: t.List[Generation] = []

        self._mutate = mutate
        self._mate = mate

        self._constraints: ConstraintSet = constraints

        self._logger: Logger = logger if logger is not None else Logger()
        self._print_log_frames = print_log_frames

        self._save_generations = save_generations

        self._mutate_threshold: float = .3
        self._mate_threshold: float = .3
        self._tournament_size: int = 4

    @property
    def logger(self) -> Logger:
        return self._logger

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

    def spawn_generation(self) -> LogFrame:
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
                        key = lambda _individual: _individual.fitness[0],
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

        if self._save_generations:
            self._generations.append(
                new_generation
            )
        else:
            self._generations = [new_generation]

        frame = self._logger.add_frame(new_generation)

        if self._print_log_frames:
            print_log_frame(len(self._logger.values) - 1, frame)

        return frame

    def spawn_generations(self, amount: int) -> Environment:
        for _ in range(amount):
            self.spawn_generation()
        return self

    def fittest(self) -> I:
        return sorted(
            self._generations[-1],
            key = lambda individual: individual.fitness,
        )[-1]
