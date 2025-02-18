This repo is forked from miguelgondu/minimal_VAE_on_Mario. See original readme further down.
All files with commit message "final commit" are new additions.
All experiments can be run with various hyperparameters by running:
```
python main.py
```
after installing requirements as described in original readme.
For the playability only model run:
```
python playabilityOnly.py
```

If you wish to run the final experiment of baseline, run:
```
python baseline_bayesian_optimization.py
```
For the final experiment of the playability constrained model, run:
```
python playability_and_jumping.py
```
To see the generated maps from the experiments look at `show_level.ipynb`


Original readme:
# A minimal example of a VAE for Mario

This folder contains three scripts: `vae.py` implements a categorical VAE with MLP encoders and decoders, `train.py` trains it, and `visualize.py` shows a snapshot of the latent space.

## Installing requirements

Create an environment using the tool of your choice. Python version `>=3.9`. Then do

```
pip install -r requirements.txt
```

## Visualizing a pretrained model

I added an already-trained model under `models/example.pt`. Run

```
python visualize.py
```

to get a look at this example's latent space.

## Using the simulator

I recently added a `simulator.jar` that lets you run levels directly from latent space. To do so, you'll need a version of Java that is above 8 (we used `OpenJDK 15.0.2`). Running

```
python simulator.py
```

should let you play content directly from latent space. Take a look at the functions implemented therein and get creative! The simulator outputs a JSON with telemetrics from the simulation, and if you set `human_player=False` it uses Robin Baumgarten's A star agent.

## Running Bayesian Optimization in latent space

I also include an example of how to run Bayesian Optimization in the latent space of the VAE. In it, I try to maximize the number of jumps. It is built using `gpytorch` and `botorch`, and you can play around with your GP definition in `simple_bayesian_optimization.py`. To run it, just call

```
python simple_bayesian_optimization.py
```

