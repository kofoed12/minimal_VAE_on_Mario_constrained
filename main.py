import sys

from consolemenu import *
from consolemenu.items import *

from baseline_bayesian_optimization import run_experiment
from playabilityOnly import run_experiment_playability_only
from playability_and_jumping import run_experiment_playability_and_jump


def run_baseline():
    pu = PromptUtils(Screen())
    # PromptUtils.input() returns an InputResult
    n_samples = int(pu.input("Number of samples").input_string)
    pu.println("\nYou entered:", n_samples, "\n")
    n_iterations = int(pu.input("Number of iterations").input_string)
    pu.println("\nYou entered:", n_iterations, "\n")
    visualize = bool(pu.input("Do you want visualization on [True/False]").input_string)
    pu.println("\nYou entered:", visualize, "\n")
    folder_name = pu.input("Name image folder").input_string
    pu.println("\nYou entered:", folder_name, "\n")
    run_experiment(visualize=visualize,n_iterations=n_iterations,n_samples=n_samples,folder_name=folder_name)
    exit()

def run_playability():
    pu = PromptUtils(Screen())
    # PromptUtils.input() returns an InputResult
    n_samples = int(pu.input("Number of samples").input_string)
    pu.println("\nYou entered:", n_samples, "\n")
    n_iterations = int(pu.input("Number of iterations").input_string)
    pu.println("\nYou entered:", n_iterations, "\n")
    visualize = bool(pu.input("Do you want visualization on [True/False]").input_string)
    pu.println("\nYou entered:", visualize, "\n")
    folder_name = pu.input("Name image folder").input_string
    pu.println("\nYou entered:", folder_name, "\n")
    run_experiment_playability_only(visualize=visualize,n_iterations=n_iterations,n_samples=n_samples,folder_name=folder_name)
    exit()

def run_playability_and_jump():
    pu = PromptUtils(Screen())
    # PromptUtils.input() returns an InputResult
    n_samples = pu.input("Number of samples").input_string
    if n_samples == '':
        n_samples = 2
        pu.println("\nWe have set number of samples to:", n_samples, "\n")
    else:
        n_samples = int(n_samples)
        pu.println("\nYou entered:", n_samples, "\n")
    n_iterations = pu.input("Number of iterations").input_string
    if n_iterations == '':
        n_iterations = 10
        pu.println("\nWe have set number of iterations to:", n_iterations, "\n")
    else:
        n_iterations = int(n_iterations)
        pu.println("\nYou entered:", n_iterations, "\n")
    visualize = pu.input("Do you want visualization on [y/n]").input_string
    if visualize != 'n':
        visualize = True
    else:
        visualize = False
    pu.println("\nVisualize is set to:", visualize, "\n")
    folder_name = pu.input("Name image folder").input_string
    if folder_name == '':
        folder_name = 'test'
        pu.println("\nWe have named your image folder:", folder_name, "\n")
    else:
        pu.println("\nYou have named your image folder:", folder_name, "\n")
    pu.println("\nYou have chosen the following parameters:")
    pu.println("\nNumber of samples:",n_samples,"\nNumber of iterations:",n_iterations,"\nVisualization:",visualize,"\nFolder name:",folder_name,"\n")
    run = pu.input("Press Enter to continue with these options. Press anything and then Enter to go back to main menu").input_string
    if run == '':
        run_experiment_playability_and_jump(visualize=visualize,n_iterations=n_iterations,n_samples=n_samples )
        exit()

def main():
    # Create the root menu
    menu = ConsoleMenu("Minimal VAE on Mario", "Choose experiment")

    # Option for running baseline version
    function_item1 = FunctionItem("Run baseline experiment", run_baseline)
    # Option for running baseline version
    function_item2 = FunctionItem("Run experiment optimizing playability", run_playability)
    # Option for running baseline version
    function_item3 = FunctionItem("Run experiment optimizing combined jumping and playability", run_playability_and_jump)

    # Add all the items to the root menu
    menu.append_item(function_item1)
    menu.append_item(function_item2)
    menu.append_item(function_item3)

    # Show the menu
    menu.start()
    menu.join()


if __name__ == "__main__":
    main()