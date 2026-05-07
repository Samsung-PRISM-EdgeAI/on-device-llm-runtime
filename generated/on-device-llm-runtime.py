import torch
import math
from typing import Tuple, Optional

"""
OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization

Source Paper: http://arxiv.org/abs/2605.04738v1

Mathematical Idea:
The core idea, "Outlier Self-Absorption" (OSA), aims to mitigate the adverse effects of
outliers in LLM weights during low-bit quantization. Outliers, which are values with
significantly larger magnitudes than the majority of weights, can disproportionately
inflate the quantization range, leading to poor precision for the more numerous
smaller values.

OSA addresses this by:
1.  **Identifying Outliers:** Values exceeding a certain threshold (e.g., a multiple of
    the standard deviation or mean absolute deviation) are flagged as outliers.
2.  **Clipping Outliers:** The identified outliers are clipped to the defined threshold.
    This reduces their extreme magnitudes.
3.  **Absorbing Residuals:** The "excess" value (the difference between the original
    outlier value and its clipped value) is calculated. This "absorbed mass"
    represents the information lost due to clipping.
4.  **Redistributing Absorbed Mass:** This "absorbed mass" is then redistributed
    among the non-outlier weights within the same quantization group. This
    redistribution is performed by adding a proportional share of the total
    absorbed mass to each non-outlier weight. This process helps to preserve
    the overall