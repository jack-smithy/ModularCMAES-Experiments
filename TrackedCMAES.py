import sys
import os
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import glob
import ioh
import shutil
from dataclasses import dataclass, fields

from modcma.modularcmaes import ModularCMAES


@dataclass
class TrackedParameters:
    sigma: float = 0
    t: int = 0
    d_norm: float = 0
    d_mean: float = 0
    ps_norm: float = 0
    ps_mean: float = 0
    pc_norm: float = 0
    pc_mean: float = 0
    lambda_: int = 0

    def update(self, parameters):
        self.sigma = parameters.sigma
        self.t = parameters.t
        self.lambda_ = parameters.lambda_

        for attr in ('D', 'ps', 'pc'):
            setattr(self, f'{attr}_norm'.lower(),
                    np.linalg.norm(getattr(parameters, attr)))
            setattr(self, f'{attr}_mean'.lower(),
                    np.mean(getattr(parameters, attr)))


class TrackedCMAES(ModularCMAES):
    def __init__(self, tracked_parameters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_parameters = tracked_parameters
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)

    def step(self):
        res = super().step()
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)
        return res


dim = 10
reps = 20

for id in range(24):

    problem = ioh.get_problem(
        fid=id+1,
        instance=1,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )

    trigger = ioh.logger.trigger.OnImprovement()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'./please-work-data/psa-fid{id+1}-{dim}D',
        root=os.getcwd(),
        algorithm_name='psa',
        store_positions=False)
    
    tracked_parameters = TrackedParameters()
        
    logger.watch(tracked_parameters, [
        x.name for x in fields(tracked_parameters)])

    problem.attach_logger(logger)

    for rep in range(reps):

        np.random.seed(rep)
        cma = TrackedCMAES(
            tracked_parameters,
            problem,
            dim,
            budget=int(1e6),
            pop_size_adaptation = 'psa'
            ).run()

        print(f'fid={id+1}/24, rep={rep+1}/{reps}, best y={problem.state.current_best.y}')

        problem.reset()
        logger.close()
