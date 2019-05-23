
import typing as t

import functools
import operator

from abc import ABC, abstractmethod


class Constraint(ABC):
	description = 'A constraint' #type: str

	def __init__(
		self,
	):
		pass

	@abstractmethod
	def score(self, individual: 'Individual') -> float:
		pass


class ConstraintSet(object):

	def __init__(self, constraints: t.Tuple[t.Tuple[Constraint, float], ...]):
		self._constraints = constraints
		self._weights = tuple(weight for _, weight in self._constraints)

	def score(self, individual: 'Individual') -> t.Tuple[float, ...]:
		unweighted_values = tuple(
			constraint.score(individual)
			for constraint, _ in
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

	def total_score(self, individual: 'Individual') -> float:
		return self.score(individual)[0]

	def __iter__(self) -> t.Iterable[Constraint]:
		return (constraint for constraint, _ in self._constraints)


class ConstraintSetBluePrint(object):

	def __init__(self, *constraints: t.Tuple[t.Type[Constraint], float, t.Dict]):
		self._constraints = constraints

	def realise(
		self,
		# constrained_nodes: t.FrozenSet[ConstrainedNode],
		# trap_amount: int,
		# random_population: t.Collection[TrapDistribution],
	) -> ConstraintSet:
		pass
		# return ConstraintSet(
		# 	tuple(
		# 		(
		# 			constraint_type(constrained_nodes, trap_amount, random_population, **kwargs),
		# 			weight,
		# 		)
		# 		for constraint_type, weight, kwargs in
		# 		self._constraints
		# 	)
		# )


class Individual(ABC):
	pass


class Population(ABC):

	@abstractmethod
	def serialize(self) -> t.AnyStr:
		pass

	@abstractmethod
	def deserialize(self, s: t.AnyStr) -> 'Population':
		pass


class DataFrame(object):
	pass


class Evolution(ABC):
	pass