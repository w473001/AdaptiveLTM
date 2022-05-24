import logging
from algorithms.fixedshare import FixedShareFactory
from algorithms.exp4 import EXP4Factory, FixedShareEXP4Factory, MarkovSpecialistsEXP4Factory
from algorithms.swarm import SwarmFactory
from algorithms.AdaptLTM import AdaptLTMFactory
from algorithms.circadianspecialists import MarkovSpecialistsFactory


logger = logging.getLogger(__name__)


class AlgoFactory:
    """
    Factory class for creating and initializing algorithms with correct parameters for simulation
    """

    def __init__(self):
        self.registry = {}

    def register(self, name: str, factory):

        if name in self.registry:
            logger.warning(f"Factory for {name} already exists. Replacing it.")
        self.registry[name] = factory

    def create(self, name: str, algo_config, global_config):
        if name not in self.registry:
            logger.warning(f"Algo {name} not in registry. Cannot create object.")
            return None

        return self.registry[name].create(algo_config, global_config)


def get_algo_factory():
    factory = AlgoFactory()
    factory.register("FIXED_SHARE", FixedShareFactory())
    factory.register("MARKOV_SPECIALISTS", MarkovSpecialistsFactory())
    factory.register("EXP4", EXP4Factory())
    factory.register("FIXED_SHARE_EXP4", FixedShareEXP4Factory())
    factory.register("MARKOV_SPECIALISTS_EXP4", MarkovSpecialistsEXP4Factory())
    factory.register("SWARM", SwarmFactory())
    factory.register("ADAPTLTM", AdaptLTMFactory())
    return factory
