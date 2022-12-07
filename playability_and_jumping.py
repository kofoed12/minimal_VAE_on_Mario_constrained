from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt
from baseline_bayesian_optimization import run_experiment

import torch as t

from torch.distributions import Uniform

import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from acquisition import ExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood
from general_functions import run_first_samples

from simulator import test_level_from_int_tensor
from vae import VAEMario
from bo_visualization_utils import plot_prediction, plot_acquisition


ROOT_DIR = Path(__file__).parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)

def bayesian_optimization_iteration_playability_and_jump(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    playabilities: t.Tensor,
    iteration: int = 0,
    plot_latent_space: bool = False,
    img_save_folder: Path = None,
    visualize: bool = False,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    # Load the model
    vae = VAEMario()
    vae.load_state_dict(t.load("./models/example.pt"))
    vae.eval()

    kernel = gpytorch.kernels.MaternKernel()
    model = SingleTaskGP(latent_codes, (jumps / 10.0), covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    acq_function = ExpectedImprovement(model, (jumps / 10.0).max())

    c_kernel = gpytorch.kernels.MaternKernel()
    c_model = SingleTaskGP(latent_codes, playabilities, covar_module=c_kernel)
    c_mll = ExactMarginalLogLikelihood(c_model.likelihood, c_model)
    fit_gpytorch_model(c_mll)
    c_model.eval()
    c_acq_function = ExpectedImprovement(c_model, playabilities.max())

    # Optimizing the acq. function by hand on a discrete grid.
    zs = t.Tensor(
        [
            [x, y]
            for x in t.linspace(-5, 5, 100)
            for y in reversed(t.linspace(-5, 5, 100))
        ]
    )
    acq_values = acq_function(zs.unsqueeze(1))
    c_acq_values = c_acq_function(zs.unsqueeze(1))
    c_acq_values_norm = (c_acq_values - c_acq_values.min()) / (c_acq_values.max() - c_acq_values.min())
    tot_acq_values = t.mul(acq_values,c_acq_values_norm)
    candidate = zs[tot_acq_values.argmax()]
    predicted_playability_probability = c_acq_values_norm[tot_acq_values.argmax()]

    level = vae.decode(candidate).probs.argmax(dim=-1)
    #print(level)
    results = test_level_from_int_tensor(level[0], visualize=visualize)

    if plot_latent_space:
        fig, (ax, ax_acq) = plt.subplots(1, 2)
        plot_prediction(c_model, ax)
        plot_acquisition(c_acq_function, ax_acq)

        ax.scatter(
            latent_codes[:, 0].cpu().detach().numpy(),
            latent_codes[:, 1].cpu().detach().numpy(),
            c="black",
            marker="x",
        )
        ax.scatter(
            [candidate[0].cpu().detach().numpy()],
            [candidate[1].cpu().detach().numpy()],
            c="red",
            marker="d",
        )

        ax.scatter(
            latent_codes[:, 0].cpu().detach().numpy(),
            latent_codes[:, 1].cpu().detach().numpy(),
            c="black",
            marker="x",
        )
        ax.scatter(
            [candidate[0].cpu().detach().numpy()],
            [candidate[1].cpu().detach().numpy()],
            c="red",
            marker="d",
        )

        if img_save_folder is not None:
            img_save_folder.mkdir(exist_ok=True, parents=True)
            fig.tight_layout()
            fig.savefig(img_save_folder / f"{iteration:04d}.png")
        # plt.show()
        plt.close(fig)

    return (
        candidate,
        t.Tensor([[results["marioStatus"]]]),
        t.Tensor([[results["jumpActionsPerformed"]]]),
        predicted_playability_probability
    )


def run_experiment_playability_and_jump(n_iterations: int = 50, visualize: bool = False):
    # Load the model
    vae = VAEMario()
    vae.load_state_dict(t.load("./models/example.pt"))
    vae.eval()

    # Get some first samples and save them.
    latent_codes, playabilities, jumps = run_first_samples(
        vae, n_samples=10, visualize=visualize
    )
    jumps = jumps.type(t.float64).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1).type(t.float64)
    playability_probability_predictions = playabilities

    # The path to save the images in.
    img_save_folder = (
        ROOT_DIR / "data" / "plots" / "bayesian_optimization" / "playabilityAndJump"
    )
    img_save_folder.mkdir(exist_ok=True, parents=True)

    # The B.O. loops; they might hang because of numerical instabilities.
    for i in range(n_iterations):
        candidate, playability, jump, playability_probability_prediction = bayesian_optimization_iteration_playability_and_jump(
            latent_codes,
            jumps,
            playabilities,
            plot_latent_space=True,
            iteration=i,
            img_save_folder=img_save_folder,
            visualize=visualize,
        )
        print(f"(Iteration {i+1}) tested {candidate} and got {jump.item()} jumps (p_pred={playability_probability_prediction}, p={playability})")
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playabilities = t.vstack((playabilities, playability))
        playability_probability_predictions = t.vstack((playability_probability_predictions,playability_probability_prediction))

