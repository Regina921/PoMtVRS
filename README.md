# PoMtVRS: Preference-Optimized Multi-Task Vehicle Routing Solver with Preference Gating

The PyTorch Implementation of ICML 2026 --

Multi-task vehicle routing solvers via deep reinforcement learning have attracted broad attention and achieved significant progress in handling multiple constraints. However, existing neural solvers still face critical challenges, including insufficient representation, unstable training, and inefficient exploration in large combinatorial action spaces, which often prevents performance from meeting its full potential. 
To address these issues, we propose PoMtVRS (Preference-Optimized Multi-Task Vehicle Routing Solver with Preference Gating), a plug-and-play framework that jointly improves decoder representations and exploration efficiency through a synergistic combination of decoder-side augmentation and preference-driven optimization. 
Specifically, we introduce the preference optimization objective to learn relative comparisons among candidate solutions for different routing tasks, encouraging a higher generation probability of better solutions. Meanwhile, we design a preference-gated block that adaptively modulates decoder representations via sparse gated attention and nonlinear residual refinement. Extensive experiments demonstrate that PoMtVRS elevates state-of-the-art unified neural VRP backbones, achieving leading performance in multi-task benchmarks and stronger generalization. 

## Overview


## Dependencies



## Download datasets and models


## Citation

## Acknowledgments
