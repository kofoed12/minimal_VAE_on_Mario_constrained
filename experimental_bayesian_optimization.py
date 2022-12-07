"""
Implements vanilla Bayesian Optimization (without constraints)
in the latent space of our SMB VAEs. The objective function
is the number of jumps thrown by the simulator (divided by ten).

After running this, you can see the latent space queries at
./data/plots/bayesian_optimization/vanilla_bo.
"""
import sys
from pathlib import Path
import gpytorch

from baseline_bayesian_optimization import run_experiment
from playabilityOnly import run_experiment_playability_only
from playability_and_jumping import run_experiment_playability_and_jump

ROOT_DIR = Path(__file__).parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)

if __name__ == "__main__":
    # visualize == seeing the agent playing on screen.
    print("arguments",sys.argv)
    if len(sys.argv) > 1:
        n_iterations = 50
        visualize = True
        if len(sys.argv) > 2:
            n_iterations = int(sys.argv[2])
        if len(sys.argv) > 3:
            visualize = bool(sys.argv[3])
        if sys.argv[1] == 'playability':
            if len(sys.argv) > 4:
                run_experiment_playability_only(visualize=visualize,n_iterations=n_iterations,folder_name=sys.argv[4])
            else:
                run_experiment_playability_only(visualize=visualize,n_iterations=n_iterations)
        elif sys.argv[1] == 'playAndJump':
            run_experiment_playability_and_jump(visualize=visualize,n_iterations=n_iterations)
        else:
            run_experiment(visualize=visualize,n_iterations=n_iterations)

    else:
        run_experiment(visualize=True,n_iterations=50)