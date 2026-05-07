import random
import torch
from typing import List, Tuple, Dict, Any, Protocol

# --- Module-level Docstring ---
"""
EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers

Source Paper: https://www.semanticscholar.org/paper/a9c75cf664f675a1b4034b0256ec3c5168e293df

This module implements the EvoPrompt framework, a novel approach for discrete prompt
optimization. It combines the generative capabilities of Large Language Models (LLMs)
with the search efficiency of Evolutionary Algorithms (EAs) to automatically discover
high-performing prompts for various tasks.

The core mathematical idea is to treat prompt optimization as a search problem in a
discrete space. An evolutionary algorithm maintains a 'population' of candidate prompts.
In each 'generation', these prompts are evaluated for their 'fitness' (how well they
perform on a given task). The best-performing prompts are selected as 'parents'.
Instead of traditional genetic operators like crossover and mutation (which are hard
to define meaningfully for natural language prompts), EvoPrompt leverages an LLM
to perform 'mutation'. The LLM is prompted to refine, rephrase, or otherwise modify
existing prompts, generating new 'offspring' prompts. These offspring then form the
next generation, and the process repeats. This iterative refinement allows the system
to explore the prompt space effectively and converge towards optimal prompts.

Key Hyperparameters and their default values:
- `population_size` (int): The number of candidate prompts maintained in each generation. Default: 10.
- `num_generations` (int): The total number of evolutionary steps to perform. Default: 5.
- `num_elite` (int): The number of best-performing prompts from the current generation
  that are directly carried over to the next generation without modification (elitism). Default: 1.
- `mutation_rate` (float): The probability that a selected parent undergoes mutation.
  (Note: In this implementation, we ensure `population_size - num_elite` new children
  are generated, so `mutation_rate` implicitly controls how many parents are selected
  to produce children, or how many children each parent produces. For simplicity,
  we ensure enough children are generated to fill the population, often by having
  each selected parent produce one child.) Default: 1.0 (all selected parents mutate).
- `selection_strategy` (str): The method used to select parents for mutation.
  Currently supports 'truncation'. Default: 'truncation'.
- `truncation_ratio` (float): If `selection_strategy` is 'truncation', this is the
  fraction of the top-performing individuals that are considered for selection. Default: 0.5.
"""

# --- Abstract Interfaces / Protocols ---

class LLM(Protocol):
    """
    Protocol for a Large Language Model (LLM) interface.
    This defines the expected methods for an LLM used in EvoPrompt.
    """
    def generate(self, prompt_template: str, **kwargs: Any) -> str:
        """
        Generates a response based on the given prompt template.

        Parameters
        ----------
        prompt_template : str
            The input prompt template for the LLM.
        **kwargs : Any
            Additional keyword arguments specific to the LLM implementation
            (e.g., temperature, max_tokens).

        Returns
        -------
        str
            The generated text response from the LLM.
        """
        ...

class FitnessFunction(Protocol):
    """
    Protocol for a fitness function interface.
    This defines the expected methods for evaluating the quality of a prompt.
    """
    def evaluate(self, prompt: str, **kwargs: Any) -> float:
        """
        Evaluates the fitness of a given prompt. Higher values indicate better fitness.

        Parameters
        ----------
        prompt : str
            The prompt string to evaluate.
        **kwargs : Any
            Additional keyword arguments specific to the fitness function
            (e.g., validation dataset, task context).

        Returns
        -------
        float
            The fitness score of the prompt.
        """
        ...

# --- Mock Implementations for Example Usage ---

class MockLLM(LLM):
    """
    A mock LLM implementation for demonstration purposes.
    It simulates prompt generation and mutation by simple string manipulation.
    """
    def __init__(self, seed: int = 42):
        """
        Initializes the MockLLM.

        Parameters
        ----------
        seed : int, optional
            Seed for random operations to ensure reproducibility, by default 42.
        """
        random.seed(seed)
        self.mutation_pool = [
            "Make it more concise.",
            "Add a call to action.",
            "Rephrase for clarity.",
            "Include an example.",
            "Focus on the user's benefit.",
            "Shorten it significantly.",
            "Expand on the details.",
            "Use more formal language.",
            "Use more informal language.",
            "Add a question at the end."
        ]

    def generate(self, prompt_template: str, **kwargs: Any) -> str:
        """
        Generates a response by applying a simple modification to the input
        or filling a template.

        If `original_prompt` is in kwargs, it simulates a mutation.
        Otherwise, it simulates initial prompt generation.

        Parameters
        ----------
        prompt_template : str
            The input prompt template.
        **kwargs : Any
            Can include `original_prompt` for mutation simulation.

        Returns
        -------
        str
            The simulated generated prompt.
        """
        if "original_prompt" in kwargs:
            original_prompt = kwargs["original_prompt"]
            # Simulate LLM mutation: pick a random mutation instruction
            mutation_instruction = random.choice(self.mutation_pool)
            # For a mock, we'll just append the instruction or slightly modify
            # A real LLM would process the instruction and original_prompt
            # to produce a new prompt.
            if "Refine the following prompt" in prompt_template:
                # Simple mock: append a random instruction and a placeholder change
                modified_prompt = f"{original_prompt} (refined: {mutation_instruction})"
                if random.random() < 0.3: # Simulate some actual change
                    words = modified_prompt.split()
                    if len(words) > 3:
                        idx = random.randint(1, len(words) - 2)
                        words[idx] = words[idx].upper() # Simple modification
                    modified_prompt = " ".join(words)
                return modified_prompt
            else:
                # Fallback for other templates, just return the template itself
                return prompt_template.format(original_prompt=original_prompt)
        else:
            # Simulate initial prompt generation
            if "initial_topic" in kwargs:
                return f"Please provide a detailed explanation of {kwargs['initial_topic']}."
            return "Generate a concise summary of the provided text." # Default initial prompt

class MockFitnessFunction(FitnessFunction):
    """
    A mock fitness function for demonstration purposes.
    It evaluates prompts based on length and presence of specific keywords.
    """
    def __init__(self, target_keywords: List[str], max_length: int = 100):
        """
        Initializes the MockFitnessFunction.

        Parameters
        ----------
        target_keywords : List[str]
            A list of keywords that increase a prompt's fitness if present.
        max_length : int, optional
            The ideal maximum length for a prompt. Prompts too long might be penalized,
            too short might also be penalized. By default 100.
        """
        self.target_keywords = [kw.lower() for kw in target_keywords]
        self.max_length = max_length

    def evaluate(self, prompt: str, **kwargs: Any) -> float:
        """
        Evaluates the fitness of a prompt based on its length and keyword presence.
        Higher fitness for prompts that are closer to `max_length` and contain
        more `target_keywords`.

        Parameters
        ----------
        prompt : str
            The prompt string to evaluate.
        **kwargs : Any
            Additional context (not used in this mock).

        Returns
        -------
        float
            The calculated fitness score.
        """
        score = 0.0
        prompt_lower = prompt.lower()

        # Reward for target keywords
        for keyword in self.target_keywords:
            if keyword in prompt_lower:
                score += 10.0

        # Reward for length, penalize for being too far from max_length
        length_diff = abs(len(prompt) - self.max_length)
        score -= length_diff * 0.1 # Penalize by 0.1 for each character away from ideal length

        # Ensure a minimum score
        return max(1.0, score)

# --- Core EvoPrompt Implementation ---

class EvoPromptOptimizer:
    """
    The EvoPromptOptimizer class implements the core evolutionary algorithm
    for prompt optimization.

    It uses an LLM for prompt generation and mutation, and a fitness function
    to evaluate prompt quality.
    """
    def __init__(
        self,
        llm: LLM,
        fitness_function: FitnessFunction,
        population_size: int = 10,
        num_generations: int = 5,
        num_elite: int = 1,
        mutation_rate: float = 1.0,
        selection_strategy: str = 'truncation',
        truncation_ratio: float = 0.5,
        seed: int = 42
    ):
        """
        Initializes the EvoPromptOptimizer.

        Parameters
        ----------
        llm : LLM
            An instance of an LLM (or mock LLM) conforming to the LLM protocol.
        fitness_function : FitnessFunction
            An instance of a fitness function (or mock fitness function)
            conforming to the FitnessFunction protocol.
        population_size : int, optional
            The number of candidate prompts in each generation, by default 10.
        num_generations : int, optional
            The total number of evolutionary generations, by default 5.
        num_elite : int, optional
            Number of top prompts carried over directly (elitism), by default 1.
        mutation_rate : float, optional
            Probability of a selected parent mutating, by default 1.0.
            (Currently, this ensures enough children are generated to fill the population).
        selection_strategy : str, optional
            Strategy for parent selection ('truncation' supported), by default 'truncation'.
        truncation_ratio : float, optional
            Fraction of top individuals considered for truncation selection, by default 0.5.
        seed : int, optional
            Seed for random operations to ensure reproducibility, by default 42.

        Raises
        ------
        ValueError
            If `num_elite` is greater than or equal to `population_size`.
            If `truncation_ratio` is not between 0 and 1.
            If an unsupported `selection_strategy` is provided.
        """
        if num_elite >= population_size:
            raise ValueError("num_elite must be less than population_size.")
        if not (0 < truncation_ratio <= 1):
            raise ValueError("truncation_ratio must be between 0 and 1.")
        if selection_strategy not in ['truncation']:
            raise ValueError(f"Unsupported selection_strategy: {selection_strategy}")

        self.llm = llm
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_elite = num_elite
        self.mutation_rate = mutation_rate
        self.selection_strategy = selection_strategy
        self.truncation_ratio = truncation_ratio
        random.seed(seed)
        torch.manual_seed(seed) # For potential future torch usage

        self.best_prompt_overall: str = ""
        self.best_fitness_overall: float = -float('inf')

    def _initialize_population(self, initial_prompt: str) -> List[str]:
        """
        Initializes the first generation of prompts.

        Parameters
        ----------
        initial_prompt : str
            A base prompt to start the optimization from.

        Returns
        -------
        List[str]
            A list of initial candidate prompts.
        """
        population = [initial_prompt]
        # Use LLM to generate variations for the rest of the population
        for _ in range(self.population_size - 1):
            # A simple way to get variations: ask LLM to rephrase the initial prompt
            # In a real scenario, this might involve more sophisticated initial sampling
            variation_prompt = self.llm.generate(
                prompt_template="Rephrase the following prompt: {original_prompt}",
                original_prompt=initial_prompt
            )
            population.append(variation_prompt)
        return population

    def _evaluate_population(self, population: List[str]) -> List[Tuple[float, str]]:
        """
        Evaluates the fitness of each prompt in the current population.

        Parameters
        ----------
        population : List[str]
            The list of prompts to evaluate.

        Returns
        -------
        List[Tuple[float, str]]
            A list of (fitness_score, prompt) tuples, sorted by fitness in descending order.
        """
        evaluated_population = []
        for prompt in population:
            fitness = self.fitness_function.evaluate(prompt)
            evaluated_population.append((fitness, prompt))
            # Track the overall best prompt
            if fitness > self.best_fitness_overall:
                self.best_fitness_overall = fitness
                self.best_prompt_overall = prompt
        
        # Sort by fitness in descending order
        evaluated_population.sort(key=lambda x: x[0], reverse=True)
        return evaluated_population

    def _select_parents(self, evaluated_population: List[Tuple[float, str]]) -> List[str]:
        """
        Selects parents from the evaluated population for mutation.

        Parameters
        ----------
        evaluated_population : List[Tuple[float, str]]
            The population with their fitness scores, sorted by fitness.

        Returns
        -------
        List[str]
            A list of selected parent prompts.
        """
        num_to_select = self.population_size - self.num_elite # Number of children needed

        if self.selection_strategy == 'truncation':
            # Truncation selection: take the top N individuals
            num_candidates = max(1, int(len(evaluated_population) * self.truncation_ratio))
            top_candidates = [p for _, p in evaluated_population[:num_candidates]]
            
            # Sample with replacement from top candidates to get enough parents
            parents = random.choices(top_candidates, k=num_to_select)
            return parents
        else:
            # This case should be caught by the constructor's validation
            raise NotImplementedError(f"Selection strategy '{self.selection_strategy}' not implemented.")

    def _mutate_prompts(self, parents: List[str]) -> List[str]:
        """
        Generates new prompts by mutating selected parents using the LLM.

        Parameters
        ----------
        parents : List[str]
            A list of parent prompts to mutate.

        Returns
        -------
        List[str]
            A list of new, mutated prompts (offspring).
        """
        offspring = []
        mutation_prompt_template = "Refine the following prompt to make it more effective: '{original_prompt}'"
        
        for parent_prompt in parents:
            # Use LLM to generate a mutated version of the parent prompt
            mutated_prompt = self.llm.generate(
                prompt_template=mutation_prompt_template,
                original_prompt=parent_prompt
            )
            offspring.append(mutated_prompt)
        return offspring

    def optimize_prompt(self, initial_prompt: str, **kwargs: Any) -> str:
        """
        Executes the EvoPrompt optimization process to find an optimal prompt.
        This function targets the `on-device-llm-runtime` by providing it
        with a highly optimized prompt for better performance.

        Parameters
        ----------
        initial_prompt : str
            The starting prompt for the optimization process.
        **kwargs : Any
            Additional keyword arguments to pass to the fitness function
            during evaluation (e.g., specific task data).

        Returns
        -------
        str
            The best-performing prompt found during the optimization process.
        """
        # Initialize population
        current_population = self._initialize_population(initial_prompt)
        self.best_prompt_overall = initial_prompt # Initialize with the initial prompt
        self.best_fitness_overall = self.fitness_function.evaluate(initial_prompt, **kwargs)

        print(f"Starting EvoPrompt optimization for initial prompt: '{initial_prompt}'")

        for generation in range(self.num_generations):
            print(f"\n--- Generation {generation + 1}/{self.num_generations} ---")

            # 1. Evaluate current population
            evaluated_population = self._evaluate_population(current_population)
            
            # Log top prompt of current generation
            current_gen_best_fitness, current_gen_best_prompt = evaluated_population[0]
            print(f"  Best prompt in current generation (fitness: {current_gen_best_fitness:.2f}): '{current_gen_best_prompt}'")
            print(f"  Overall best prompt so far (fitness: {self.best_fitness_overall:.2f}): '{self.best_prompt_overall}'")

            # 2. Elitism: Carry over the best individuals directly
            next_population = [prompt for _, prompt in evaluated_population[:self.num_elite]]

            # 3. Select parents for mutation
            parents_for_mutation = self._select_parents(evaluated_population)

            # 4. Mutate parents to create offspring
            offspring = self._mutate_prompts(parents_for_mutation)

            # 5. Form the new population
            # Ensure the new population size matches `self.population_size`
            # If `len(offspring)` is less than `population_size - num_elite`,
            # we might need to generate more or resample.
            # For simplicity, we assume `_select_parents` and `_mutate_prompts`
            # together produce `population_size - num_elite` offspring.
            next_population.extend(offspring)
            
            # Trim or pad if necessary (shouldn't be needed if selection is correct)
            if len(next_population) > self.population_size:
                next_population = next_population[:self.population_size]
            elif len(next_population) < self.population_size:
                # This case indicates an issue in selection/mutation logic for population size
                # For robustness, fill with random existing prompts or more mutations
                print(f"  Warning: New population size ({len(next_population)}) is less than target ({self.population_size}). Padding.")
                while len(next_population) < self.population_size:
                    next_population.append(random.choice(current_population)) # Simple padding

            current_population = next_population

        print(f"\n--- Optimization Complete ---")
        print(f"Final best prompt found (fitness: {self.best_fitness_overall:.2f}): '{self.best_prompt_overall}'")
        return self.best_prompt_overall

if __name__ == "__main__":
    print("--- EvoPrompt Example Usage for on-device-llm-runtime ---")
    print("This example demonstrates how EvoPrompt can optimize a prompt for a hypothetical")
    print("on-device LLM task, aiming for clarity and conciseness related to 'battery life'.")

    # 1. Instantiate Mock LLM and Fitness Function
    # The MockLLM simulates an actual LLM that would perform prompt generation/mutation.
    # The MockFitnessFunction simulates evaluating a prompt's effectiveness for a task.
    mock_llm = MockLLM(seed=100)
    
    # For our on-device LLM, we might want prompts that are concise, clear,
    # and mention key aspects like "efficiency" or "performance" for a task
    # related to device optimization.
    target_keywords_for_on_device_task = ["efficiency", "performance", "battery life", "optimize"]
    mock_fitness_func = MockFitnessFunction(
        target_keywords=target_keywords_for_on_device_task,
        max_length=80 # On-device LLMs might prefer shorter prompts
    )

    # 2. Instantiate EvoPromptOptimizer
    # We'll use a small population and few generations for a quick example.
    # In a real scenario, these would be larger.
    optimizer = EvoPromptOptimizer(
        llm=mock_llm,
        fitness_function=mock_fitness_func,
        population_size=8,
        num_generations=3,
        num_elite=1,
        truncation_ratio=0.7,
        seed=200
    )

    # 3. Define an initial prompt for the on-device LLM task
    # Let's say the task is to summarize device status.
    initial_prompt_for_on_device_llm = "Summarize the current device status and power consumption."

    print(f"\nInitial prompt: '{initial_prompt_for_on_device_llm}'")
    initial_fitness = mock_fitness_func.evaluate(initial_prompt_for_on_device_llm)
    print(f"Initial prompt fitness: {initial_fitness:.2f}")

    # 4. Run the optimization
    # The `optimize_prompt` method returns the best prompt found.
    # This optimized prompt would then be used by the `on-device-llm-runtime`.
    optimized_prompt = optimizer.optimize_prompt(initial_prompt_for_on_device_llm)

    print("\n--- Optimization Results ---")
    print(f"Original Prompt: '{initial_prompt_for_on_device_llm}'")
    print(f"Optimized Prompt: '{optimized_prompt}'")
    print(f"Original Fitness: {initial_fitness:.2f}")
    print(f"Optimized Fitness: {mock_fitness_func.evaluate(optimized_prompt):.2f}")

    # Example of how the `on-device-llm-runtime` would use this:
    # Imagine `on_device_llm_runtime.run(prompt, input_data)`
    # Instead of:
    #   response_original = on_device_llm_runtime.run(initial_prompt_for_on_device_llm, device_data)
    # It would use the optimized one:
    #   response_optimized = on_device_llm_runtime.run(optimized_prompt, device_data)
    print("\n--- Simulating on-device-llm-runtime usage ---")
    print(f"The `on-device-llm-runtime` would now use the optimized prompt:")
    print(f"  `on_device_llm_runtime.run(optimized_prompt, device_sensor_data)`")
    print(f"  Optimized prompt: '{optimized_prompt}'")

    # Another example with different initial prompt and keywords
    print("\n--- Another EvoPrompt Example: Customer Support Prompt ---")
    mock_llm_2 = MockLLM(seed=300)
    target_keywords_customer_support = ["solution", "issue", "resolve", "kindly", "assist"]
    mock_fitness_func_2 = MockFitnessFunction(
        target_keywords=target_keywords_customer_support,
        max_length=120
    )
    optimizer_2 = EvoPromptOptimizer(
        llm=mock_llm_2,
        fitness_function=mock_fitness_func_2,
        population_size=10,
        num_generations=4,
        num_elite=2,
        truncation_ratio=0.6,
        seed=400
    )
    initial_prompt_customer_support = "Help me with my problem."
    print(f"\nInitial prompt: '{initial_prompt_customer_support}'")
    initial_fitness_2 = mock_fitness_func_2.evaluate(initial_prompt_customer_support)
    print(f"Initial prompt fitness: {initial_fitness_2:.2f}")

    optimized_prompt_2 = optimizer_2.optimize_prompt(initial_prompt_customer_support)

    print("\n--- Optimization Results (Customer Support) ---")
    print(f"Original Prompt: '{initial_prompt_customer_support}'")
    print(f"Optimized Prompt: '{optimized_prompt_2}'")
    print(f"Original Fitness: {initial_fitness_2:.2f}")
    print(f"Optimized Fitness: {mock_fitness_func_2.evaluate(optimized_prompt_2):.2f}")