import sys
from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt
from baseline_bayesian_optimization import run_experiment

import torch as t

from torch.distributions import Uniform

import gpytorch

from botorch.models import SingleTaskGP
from general_functions import run_first_samples
from simulator import test_level_from_int_tensor
from vae import VAEMario
from bo_visualization_utils import plot_prediction, plot_probability

from GPClassificationModel import GPClassificationModel, train_classification_model


ROOT_DIR = Path(__file__).parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)


def bayesian_optimization_iteration_only_playability(
    latent_codes: t.Tensor,
    #jumps: t.Tensor,
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

    playabilities = playabilities.type(t.float)
    cf_model = GPClassificationModel(latent_codes)
    cf_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    cf_model.train()
    cf_likelihood.train()

    train_classification_model(cf_model,cf_likelihood,latent_codes,playabilities.squeeze())
    cf_model.eval()
    cf_likelihood.eval()

    # Optimizing the acq. function by hand on a discrete grid.
    zs = t.Tensor(
        [
            [x, y]
            for x in t.linspace(-5, 5, 100)
            for y in reversed(t.linspace(-5, 5, 100))
        ]
    )
    cf_preds = cf_likelihood(cf_model(zs.unsqueeze(1).type(t.float)))
    pred_labels = cf_preds.mean
    candidate = zs[pred_labels.argmax()]
    predicted_playability_probability = pred_labels[pred_labels.argmax()]
    print("candidate",candidate,"predicted_playability_probability",predicted_playability_probability)

    level = vae.decode(candidate).probs.argmax(dim=-1)
    #print(level)
    results = test_level_from_int_tensor(level[0], visualize=visualize)

    if plot_latent_space:
        fig, (ax, ax_acq) = plt.subplots(1, 2)
        plot_prediction(cf_model, ax)
        plot_probability(pred_labels, ax_acq)

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


def run_experiment_playability_only(n_iterations: int = 50, visualize: bool = False, folder_name: str = 'playability_only'):
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
        ROOT_DIR / "data" / "plots" / "bayesian_optimization" / folder_name
    )
    img_save_folder.mkdir(exist_ok=True, parents=True)

    # The B.O. loops; they might hang because of numerical instabilities.
    for i in range(n_iterations):
        candidate, playability, jump, playability_probability_prediction = bayesian_optimization_iteration_only_playability(
            latent_codes,
            playabilities,
            plot_latent_space=True,
            iteration=i,
            img_save_folder=img_save_folder,
            visualize=visualize,
        )
        print(f"(Iteration {i+1}) tested {candidate} and got {jump.item()} jumps (p_pred={playability_probability_prediction}, p={playability})")
        # if playability == 0.0:
        #     jump = t.zeros_like(jump)
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playabilities = t.vstack((playabilities, playability))
        playability_probability_predictions = t.vstack((playability_probability_predictions,playability_probability_prediction))
