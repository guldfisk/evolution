from __future__ import annotations

import typing as t
import random
import copy

from functools import reduce
from operator import add
from abc import abstractmethod

from evolution.logging import Logger, LogFrame, print_log_frame
from evolution.model import Individual, ConstraintSet, Generation


I = t.TypeVar('I', bound = Individual)


class EvolutionModel(t.Generic[I]):

    def __init__(
        self,
        mutate: t.Callable[[I, Environment], I],
        mate: t.Callable[[I, I, Environment], t.Tuple[I, I]],
        individual_factory: t.Callable[[], I],
        constraints: ConstraintSet,
    ):
        self._mutate = mutate
        self._mate = mate
        self._individual_factory = individual_factory
        self._constraints = constraints
        self._environment: t.Optional[Environment] = None

    def _set_environment(self, environment: Environment) -> None:
        self._environment = environment

    environment = property(fset = _set_environment)

    def mutate_population(self, generation: Generation, threshold: float) -> None:
        for individual in generation:
            if random.random() < threshold:
                self._mutate(individual, self._environment)
                setattr(individual, '_changed', True)

    def mate_population(self, generation: Generation, threshold: float) -> None:
        for first, second in zip(
            generation[0::2],
            generation[1::2],
        ):
            if random.random() < threshold:
                self._mate(first, second, self._environment)
                setattr(first, '_changed', True)
                setattr(second, '_changed', True)

    @abstractmethod
    def get_first_generation(self) -> Generation:
        pass

    @abstractmethod
    def get_next_generation(self) -> Generation:
        pass


class SimpleModel(EvolutionModel[I]):

    def __init__(
        self,
        mutate: t.Callable[[I, Environment], I],
        mate: t.Callable[[I, I, Environment], t.Tuple[I, I]],
        individual_factory: t.Callable[[], I],
        initial_population_size: int,
        constraints: ConstraintSet,
    ):
        super().__init__(
            mutate,
            mate,
            individual_factory,
            constraints,
        )
        self._initial_population_size = initial_population_size

        self._mutate_threshold: float = .3
        self._mate_threshold: float = .3
        self._tournament_size: int = 4

        self._generation: t.Optional[Generation] = None


    def get_first_generation(self) -> Generation:
        generation = [
            self._individual_factory()
            for _ in
            range(self._initial_population_size)
        ]
        for individual in generation:
            setattr(
                individual,
                '_fitness',
                individual.score(self._constraints)
            )

        self._generation = generation

        return generation

    def get_next_generation(self) -> Generation:
        new_generation = [
            copy.deepcopy(
                max(
                    random.sample(
                        self._generation,
                        self._tournament_size,
                    ),
                    key = lambda _individual: _individual.fitness[0],
                )
            )
            for _ in
            range(self._initial_population_size)
        ]

        for individual in new_generation:
            if hasattr(individual, '_changed'):
                delattr(individual, '_changed')

        self.mate_population(new_generation, self._mate_threshold)
        self.mutate_population(new_generation, self._mutate_threshold)

        for individual in new_generation:
            if hasattr(individual, '_changed'):
                setattr(
                    individual,
                    '_fitness',
                    individual.score(self._constraints)
                )

        self._generation = new_generation

        return new_generation


class VolatileGroupModel(EvolutionModel[I]):

    def __init__(
        self,
        mutate: t.Callable[[I, Environment], I],
        mate: t.Callable[[I, I, Environment], t.Tuple[I, I]],
        individual_factory: t.Callable[[], I],
        stable_population_size: int,
        volatile_population_size: int,
        constraints: ConstraintSet,
    ):
        super().__init__(mutate, mate, individual_factory, constraints)

        self._stable_population_size = stable_population_size
        self._volatile_population_size = volatile_population_size

        self._stable_from_volatile = self._stable_population_size // 4
        self._stable_from_stable = self._stable_population_size - self._stable_from_volatile

        self._volatile_from_stable = self._volatile_population_size // 4
        self._volatile_from_volatile = self._volatile_population_size - self._volatile_from_stable

        self._stable_mutate_threshold: float = .2
        self._stable_mate_threshold: float = .3

        self._volatile_mutate_threshold: float = .7
        self._volatile_mate_threshold: float = .2

        self._stable_tournament_size: int = 4
        self._volatile_tournament_size: int = 2

        self._stable: t.List[I] = []
        self._volatile: t.List[I] = []

    def get_first_generation(self) -> Generation:
        self._stable = [
            self._individual_factory()
            for _ in
            range(self._stable_population_size)
        ]
        self._volatile = [
            self._individual_factory()
            for _ in
            range(self._volatile_population_size)
        ]

        generation = self._stable + self._volatile

        for individual in generation:
            setattr(
                individual,
                '_fitness',
                individual.score(
                    self._constraints
                )
            )

        return generation

    def get_next_generation(self) -> Generation:
        new_stable = [
            copy.deepcopy(
                max(
                    random.sample(
                        self._stable,
                        self._stable_tournament_size,
                    ),
                    key = lambda _individual: _individual.fitness[0],
                )
            )
            for _ in
            range(self._stable_from_stable)
        ] + [
            copy.deepcopy(
                max(
                    random.sample(
                        self._volatile,
                        self._stable_tournament_size,
                    ),
                    key = lambda _individual: _individual.fitness[0],
                )
            )
            for _ in
            range(self._stable_from_volatile)
        ]
        random.shuffle(new_stable)

        new_volatile = [
            copy.deepcopy(
                max(
                    random.sample(
                        self._volatile,
                        self._volatile_tournament_size,
                    ),
                    key = lambda _individual: _individual.fitness[0],
                )
            )
            for _ in
            range(self._volatile_from_volatile)
        ] + [
            copy.deepcopy(
                max(
                    random.sample(
                        self._stable,
                        self._stable_tournament_size,
                    ),
                    key = lambda _individual: _individual.fitness[0],
                )
            )
            for _ in
            range(self._volatile_from_stable)
        ]
        random.shuffle(new_volatile)

        new_generation = new_stable + new_volatile

        for individual in new_generation:
            if hasattr(individual, '_changed'):
                delattr(individual, '_changed')

        self.mate_population(new_stable, self._stable_mate_threshold)
        self.mutate_population(new_stable, self._stable_mutate_threshold)

        self.mate_population(new_volatile, self._volatile_mate_threshold)
        self.mutate_population(new_volatile, self._volatile_mutate_threshold)

        for individual in new_generation:
            if hasattr(individual, '_changed'):
                setattr(
                    individual,
                    '_fitness',
                    individual.score(
                        self._constraints
                    ),
                )

        self._stable = new_stable
        self._volatile = new_volatile

        return new_generation


class IslandModel(EvolutionModel[I]):

    def __init__(
        self,
        mutate: t.Callable[[I, Environment], I],
        mate: t.Callable[[I, I, Environment], t.Tuple[I, I]],
        individual_factory: t.Callable[[], I],
        constraints: ConstraintSet,
        island_count: int = 3,
        island_size: int = 100,
    ):
        super().__init__(mutate, mate, individual_factory, constraints)

        self._island_count = island_count
        self._island_size = island_size

        self._mutate_threshold: float = .3
        self._mate_threshold: float = .3
        self._tournament_size: int = 4

        self._islands: t.List[Generation] =  [[] for _ in range(self._island_count)]

    @property
    def _generation(self) -> Generation:
        return reduce(add, self._islands)

    def get_first_generation(self) -> Generation:
        for island in self._islands:
            island[:] = [
                self._individual_factory()
                for _ in
                range(self._island_size)
            ]

        generation = self._generation

        for individual in generation:
            setattr(
                individual,
                '_fitness',
                individual.score(
                    self._constraints
                ),
            )

        return generation

    def get_next_generation(self) -> Generation:
        new_islands = []
        for index, island in enumerate(self._islands):
            if random.random() < 1/50:
                options = island + self._islands[random.choice(list(set(range(self._island_count)) - {index}))]
            else:
                options = island
            new_population = [
                    copy.deepcopy(
                        max(
                            random.sample(
                                options,
                                self._tournament_size,
                            ),
                            key = lambda _individual: _individual.fitness[0],
                        )
                    )
                    for _ in
                    range(self._island_size)
                ]

            for individual in new_population:
                if hasattr(individual, '_changed'):
                    delattr(individual, '_changed')

            self.mate_population(new_population, self._mate_threshold)
            self.mutate_population(new_population, self._mutate_threshold)

            new_islands.append(new_population)

        self._islands = new_islands

        generation = self._generation

        for individual in generation:
            if hasattr(individual, '_changed'):
                setattr(
                    individual,
                    '_fitness',
                    individual.score(
                        self._constraints
                    ),
                )

        return generation


class Environment(t.Generic[I]):
    
    def __init__(
        self,
        model: EvolutionModel[I],
        logger: t.Optional[Logger] = None,
        *,
        save_generations: bool = True,
        print_log_frames: bool = False,
    ):
        self._model = model
        self._model.environment = self

        self._logger: Logger = logger if logger is not None else Logger()
        self._print_log_frames = print_log_frames

        self._save_generations = save_generations

        self._generations: t.List[Generation] = []
        
    @property
    def logger(self) -> Logger:
        return self._logger

    def spawn_generation(self) -> LogFrame:
        if not self._generations:
            new_generation = self._model.get_first_generation()
        else:
            new_generation = self._model.get_next_generation()

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
