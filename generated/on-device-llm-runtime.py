import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Union

# --- Module-level Docstring ---
"""
Implicit Representations of Grammaticality in Language Models

Source Paper: http://arxiv.org/abs/2605.05197v1

Mathematical Idea:
The core idea is to assign a "grammaticality score" to a candidate text sequence by
comparing its log-likelihood under a fine-tuned language model (LM) with the
average log-likelihood of several grammatically perturbed versions of that sequence.
A higher score indicates better grammaticality. This is because a truly grammatical
sentence should have a high log-likelihood (the LM is "certain" about it), while
its grammatically incorrect variants should have lower log-likelihoods (the LM
is "uncertain" about them). The difference between the original's log-likelihood
and the average log-likelihood of its perturbed versions serves as this
grammaticality score. This score can then be used to re-rank candidate generations
from an LLM, prioritizing more grammatical outputs.

Key Hyperparameters and their default values:
- num_perturbations (int, default: 5): The number of grammatically perturbed
  versions to generate for each candidate sequence.
- perturbation_strength (float, default: 0.1): The proportion of words in a
  sentence that might be affected by a perturbation (e.g., 0.1 means ~10% of words).
- perturbation_types (