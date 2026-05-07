"""
Module implementing the EvoPrompt algorithm from the paper:
"EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers"
https://www.semanticscholar.org/paper/a9c75cf664f675a1b4034b0256ec3c5168e293df

This module provides a Python implementation of the EvoPrompt algorithm, which optimizes discrete prompts using evolutionary algorithms.
The goal is to reduce the prompt length and complexity for on-device inference, enhancing tokens/sec.

Key hyperparameters:
- population_size (int, default=100): The size of the population in the evolutionary algorithm.
- mutation_rate (float, default=0.1): The rate at which the prompts are mutated.
- num_generations (int, default=100): The number of generations to run the evolutionary algorithm.
- selection_rate (float, default=0.5): The rate at which the fittest prompts are selected.

"""

import torch
import numpy as np
import random

def on_device_llm_runtime(prompt, device):
    """
    Evaluate the prompt on the on-device LLM runtime.

    Parameters
    ----------
    prompt : str
        The prompt to evaluate.
    device : torch.device
        The device to run the evaluation on.

    Returns
    -------
    score : float
        The score of the prompt.
    """
    # For demonstration purposes, we assume a simple language model that returns a random score
    return random.random()

def generate_initial_population(size, prompt_length):
    """
    Generate the initial population of prompts.

    Parameters
    ----------
    size : int
        The size of the population.
    prompt_length : int
        The length of the prompts.

    Returns
    -------
    population : list of str
        The initial population of prompts.
    """
    # Initialize an empty population
    population = []
    # Generate random prompts of the specified length
    for _ in range(size):
        prompt = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(prompt_length))
        population.append(prompt)
    return population

def evaluate_population(population, device):
    """
    Evaluate the population of prompts on the on-device LLM runtime.

    Parameters
    ----------
    population : list of str
        The population of prompts to evaluate.
    device : torch.device
        The device to run the evaluation on.

    Returns
    -------
    scores : list of float
        The scores of the prompts.
    """
    # Evaluate each prompt in the population
    scores = [on_device_llm_runtime(prompt, device) for prompt in population]
    return scores

def select_fittest(population, scores, selection_rate):
    """
    Select the fittest prompts from the population.

    Parameters
    ----------
    population : list of str
        The population of prompts to select from.
    scores : list of float
        The scores of the prompts.
    selection_rate : float
        The rate at which the fittest prompts are selected.

    Returns
    -------
    selected_population : list of str
        The selected population of prompts.
    """
    # Sort the population by score in descending order
    sorted_population = [prompt for _, prompt in sorted(zip(scores, population), reverse=True)]
    # Select the top prompts based on the selection rate
    num_selected = int(len(population) * selection_rate)
    selected_population = sorted_population[:num_selected]
    return selected_population

def mutate_population(population, mutation_rate):
    """
    Mutate the population of prompts.

    Parameters
    ----------
    population : list of str
        The population of prompts to mutate.
    mutation_rate : float
        The rate at which the prompts are mutated.

    Returns
    -------
    mutated_population : list of str
        The mutated population of prompts.
    """
    # Initialize an empty mutated population
    mutated_population = []
    # Mutate each prompt in the population
    for prompt in population:
        if random.random() < mutation_rate:
            # Randomly select a character in the prompt to mutate
            mutate_index = random.randint(0, len(prompt) - 1)
            # Replace the character with a random one
            prompt = prompt[:mutate_index] + random.choice('abcdefghijklmnopqrstuvwxyz') + prompt[mutate_index + 1:]
        mutated_population.append(prompt)
    return mutated_population

def crossover_population(population, crossover_rate):
    """
    Perform crossover between the prompts in the population.

    Parameters
    ----------
    population : list of str
        The population of prompts to crossover.
    crossover_rate : float
        The rate at which the prompts are crossed over.

    Returns
    -------
    crossed_over_population : list of str
        The crossed over population of prompts.
    """
    # Initialize an empty crossed over population
    crossed_over_population = []
    # Perform crossover between each pair of prompts in the population
    for i in range(0, len(population), 2):
        if random.random() < crossover_rate:
            # Randomly select a crossover point
            crossover_point = random.randint(1, len(population[i]) - 1)
            # Perform crossover
            prompt1 = population[i][:crossover_point] + population[i + 1][crossover_point:]
            prompt2 = population[i + 1][:crossover_point] + population[i][crossover_point:]
            crossed_over_population.extend([prompt1, prompt2])
        else:
            crossed_over_population.extend([population[i], population[i + 1]])
    return crossed_over_population

def evolve_prompt(prompt_length, population_size, mutation_rate, num_generations, selection_rate, crossover_rate, device):
    """
    Evolve a prompt using the EvoPrompt algorithm.

    Parameters
    ----------
    prompt_length : int
        The length of the prompt to evolve.
    population_size : int
        The size of the population.
    mutation_rate : float
        The rate at which the prompts are mutated.
    num_generations : int
        The number of generations to run the evolutionary algorithm.
    selection_rate : float
        The rate at which the fittest prompts are selected.
    crossover_rate : float
        The rate at which the prompts are crossed over.
    device : torch.device
        The device to run the evaluation on.

    Returns
    -------
    evolved_prompt : str
        The evolved prompt.
    """
    # Initialize the population
    population = generate_initial_population(population_size, prompt_length)
    # Run the evolutionary algorithm for the specified number of generations
    for _ in range(num_generations):
        # Evaluate the population
        scores = evaluate_population(population, device)
        # Select the fittest prompts
        selected_population = select_fittest(population, scores, selection_rate)
        # Mutate the selected prompts
        mutated_population = mutate_population(selected_population, mutation_rate)
        # Perform crossover between the mutated prompts
        crossed_over_population = crossover_population(mutated_population, crossover_rate)
        # Replace the least fit prompts with the crossed over prompts
        population = selected_population + crossed_over_population[:population_size - len(selected_population)]
    # Return the fittest prompt
    scores = evaluate_population(population, device)
    return population[np.argmax(scores)]

if __name__ == "__main__":
    # Set the parameters
    prompt_length = 10
    population_size = 100
    mutation_rate = 0.1
    num_generations = 100
    selection_rate = 0.5
    crossover_rate = 0.5
    device = torch.device("cpu")
    # Evolve a prompt
    evolved_prompt = evolve_prompt(prompt_length, population_size, mutation_rate, num_generations, selection_rate, crossover_rate, device)
    print("Evolved prompt:", evolved_prompt)