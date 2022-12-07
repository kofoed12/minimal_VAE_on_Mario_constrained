from vae import VAEMario
import torch as t
from typing import Tuple
from torch.distributions import Uniform
from simulator import test_level_from_int_tensor


def run_first_samples(
    vae: VAEMario,
    n_samples: int = 10,
    visualize: bool = False,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs the simulator on {n_samples} levels selected uniformly
    at random from the latent space (considered to be bounded in
    the [-5, 5]^2 square). Returns the latent codes, jumps and
    playabilities (a binary value stating whether the level was
    solved or not).
    """
    latent_codes = Uniform(t.Tensor([-5.0, -5.0]), t.Tensor([5.0, 5.0])).sample(
        (n_samples,)
    )
    levels = vae.decode(latent_codes).probs.argmax(dim=-1)

    playability = []
    jumps = []
    for i, level in enumerate(levels):
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

    # Returning.
    return latent_codes, t.Tensor(playability), t.Tensor(jumps)
