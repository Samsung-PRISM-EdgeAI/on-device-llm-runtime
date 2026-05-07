import random
import torch
import numpy as np
import re
from typing import List, Dict, Callable, Tuple, Any

# Module-level docstring
"""
EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers
Source Paper: https://www.semanticscholar.org/paper/a9c75cf664f675a1b4034b025ec6c3c5168e293df

This module implements the EvoPrompt framework, a novel approach for discrete prompt
optimization. It combines the power of Large Language Models (LLMs) with Evolutionary
Algorithms (EAs) to iteratively refine and improve prompts for specific tasks.

The core idea is to treat prompt optimization as a search problem in the discrete
space of natural language. An evolutionary algorithm maintains a population of prompts.
In each generation:
1.  **Evaluation**: Each prompt's fitness is assessed by running it through a target
    LLM (e.g., an on-device LLM) on a given dataset and measuring its performance
    (e.g., accuracy, F1-score).
2.  **Selection**: Prompts with higher fitness are selected as "parents."
3.  **Reproduction (LLM-based Mutation/Crossover)**: A separate "prompt generator" LLM
    is used to create new "offspring" prompts. This LLM is prompted with the task
    description and examples of high-performing parent prompts, instructing it