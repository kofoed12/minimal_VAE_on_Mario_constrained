from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt

import torch as t

import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from GPClassificationModel import GPClassificationModel, train_classification_model
from acquisition import ExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood
from general_functions import run_first_samples

from simulator import test_level_from_int_tensor
from vae import VAEMario
from bo_visualization_utils import plot_prediction, plot_acquisition, plot_probability


ROOT_DIR = Path(__file__).parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)

def run_final_level(
    level: t.Tensor,
    n_samples: int = 100,
    visualize: bool = True,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs the simulator on {n_samples} levels selected uniformly
    at random from the latent space (considered to be bounded in
    the [-5, 5]^2 square). Returns the latent codes, jumps and
    playabilities (a binary value stating whether the level was
    solved or not).
    """

    playability = []
    jumps = []
    jumps_and_playability = []
    for i in range(n_samples):
        results = test_level_from_int_tensor(level, visualize=visualize)
        playability.append(results["marioStatus"])
        jumps.append(results["jumpActionsPerformed"])
        print(
            "i:",
            i,
            "p:",
            results["marioStatus"],
            "jumps:",
            results["jumpActionsPerformed"],
        )
        if results["marioStatus"] == 1:
            jumps_and_playability.append(results["jumpActionsPerformed"])
            print("jumps_cond", results["jumpActionsPerformed"])
        else:
            jumps_and_playability.append(results["marioStatus"])
            print("jumps_cond", results["marioStatus"])

    # Returning.
    return t.Tensor(playability).type(t.float64), t.Tensor(jumps), t.Tensor(jumps_and_playability)

def bayesian_optimization_final_iteration(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    playabilities: t.Tensor
) -> t.Tensor:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    # Load the model
    vae = VAEMario()
    vae.load_state_dict(t.load("./models/example.pt"))
    vae.eval()

    kernel = gpytorch.kernels.MaternKernel()
    model = SingleTaskGP(latent_codes.type(t.double), (jumps / 10.0), covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    print("best_f",jumps[playabilities==1].max())

    cf_model = GPClassificationModel(latent_codes)
    cf_likelihood= gpytorch.likelihoods.BernoulliLikelihood()

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
    predicted_means = model(zs.unsqueeze(1)).mean.squeeze(1)
    cf_preds = cf_likelihood(cf_model(zs.unsqueeze(1)))
    pred_labels = (cf_preds.mean).squeeze(1)
    # 
    expected_jumps = t.mul(predicted_means,pred_labels)
    candidate = zs[expected_jumps.argmax()]
    print("best expected jumps", expected_jumps[expected_jumps.argmax()])
    level = vae.decode(candidate).probs.argmax(dim=-1)
    return level[0]


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
    model = SingleTaskGP(latent_codes.type(t.double), (jumps / 10.0), covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    print("best_f",jumps[playabilities==1].max())
    acq_function = ExpectedImprovement(model, best_f=(jumps[playabilities==1].max() / 10.0))

    cf_model = GPClassificationModel(latent_codes)
    cf_likelihood= gpytorch.likelihoods.BernoulliLikelihood()

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
    acq_values = acq_function(zs.unsqueeze(1))
    # acq_values = (acq_values - acq_values.min()) / (acq_values.max() - acq_values.min())
    cf_preds = cf_likelihood(cf_model(zs.unsqueeze(1)))
    pred_labels = (cf_preds.mean).squeeze(1)
    # 
    tot_acq_values = t.mul(acq_values,pred_labels)
    candidate = zs[tot_acq_values.argmax()]
    candidate_acq_value = tot_acq_values[tot_acq_values.argmax()]

    level = vae.decode(candidate).probs.argmax(dim=-1).squeeze(0)
    #print(level)
    playability_it = 0
    jumps_it = 0
    for j in range(10):
        results = test_level_from_int_tensor(level, visualize=visualize)
        playability_it += results["marioStatus"]
        jumps_it += results["jumpActionsPerformed"]
    if playability_it > 5:
        playability_result = 1
    else:
        playability_result = 0
    jumps_result = jumps_it/10.0
    if plot_latent_space:
        plt.switch_backend('Agg')
        fig, (ax, ax_acq, ax_prob) = plt.subplots(1, 3)
        plot_prediction(model, ax)
        plot_acquisition(acq_function, ax_acq)
        plot_probability(pred_labels, ax_prob)


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
        t.Tensor([playability_result]),
        t.Tensor([jumps_result]),
        candidate_acq_value
    )


def run_experiment_playability_and_jump(n_iterations: int = 50, n_samples: int = 10, visualize: bool = False, folder_name: str = 'final_test_multisim'):
    # Load the model
    vae = VAEMario()
    vae.load_state_dict(t.load("./models/example.pt"))
    vae.eval()

    # Get some first samples and save them.
    latent_codes, playabilities, jumps = run_first_samples(
        vae, n_samples=n_samples, visualize=visualize
    )
    jumps = jumps.type(t.float64).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1)
    candidate_acq_values = playabilities

    # The path to save the images in.
    img_save_folder = (
        ROOT_DIR / "data" / "plots" / "bayesian_optimization" / "playabilityAndJump" / folder_name
    )
    img_save_folder.mkdir(exist_ok=True, parents=True)
    # The B.O. loops; they might hang because of numerical instabilities.
    for i in range(n_iterations):
        candidate, playability, jump, candidate_acq_value = bayesian_optimization_iteration_playability_and_jump(
            latent_codes,
            jumps,
            playabilities,
            plot_latent_space=True,
            iteration=i,
            img_save_folder=img_save_folder,
            visualize=visualize,
        )
        print("candidate shape", candidate.shape)
        print(f"(Iteration {i+1}) tested {candidate} and got {jump.item()} jumps (acq_value={candidate_acq_value}, p={playability})")
        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playabilities = t.vstack((playabilities, playability))
        candidate_acq_values = t.vstack((candidate_acq_values,candidate_acq_value))
    
    best_level = bayesian_optimization_final_iteration(latent_codes=latent_codes,jumps=jumps, playabilities=playabilities)
    print("best_level",best_level)
    final_playabilites, final_jumps, final_jumps_cond = run_final_level(best_level)
    print("expected number of jumps",final_jumps.mean(), "expected number of jumps conditioned on playability==1", final_jumps_cond.mean(), "finished", t.sum(final_playabilites), "times out of 100")


if __name__ == "__main__":
    # visualize == seeing the agent playing on screen.
    run_experiment_playability_and_jump(visualize=True, folder_name='final_test2')