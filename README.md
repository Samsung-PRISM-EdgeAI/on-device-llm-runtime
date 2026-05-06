# On-Device LLM Runtime - Repository Context

## Overview
This repository contains the runtime engine for deploying Large Language Models (LLMs) like Llama-3 or Gemma directly onto Samsung mobile devices (Galaxy S-series). 

## Tech Stack & Architecture
- **Language**: Python (Triton/Cuda simulated wrappers)
- **Models**: 3B to 8B parameter quantized models.
- **Constraints**:
  - Memory is the biggest bottleneck. The KV Cache quickly eats up RAM.
  - Token generation speed needs to be at least 15 tokens/sec for a good user experience.

## Current Goals & Challenges
1. **KV Cache Compression**: We are hitting Out-Of-Memory (OOM) errors during long conversations. We need ways to compress, page, or offload the KV cache without losing context.
2. **Decoding Speed**: We want to implement speculative decoding to predict multiple tokens at once, bypassing the memory bandwidth bottleneck of the mobile CPU/NPU.
