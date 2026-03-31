## Reinforcement learning implementation for "Chemotactic navigation in robotic swimmers via reset-free hierarchical reinforcement learning"
[![arXiv](https://img.shields.io/badge/arXiv-2408.07346-df2a2a.svg)](https://arxiv.org/pdf/2408.07346)
[![Python](https://img.shields.io/badge/python-3.7.17-blue)](https://www.python.org)
[![DOI](https://zenodo.org/badge/858608778.svg)](https://doi.org/10.5281/zenodo.15490877)


Tongzhao Xiong, Zhaorong Liu, Yufei Wang, Chong Jin Ong, Lailai Zhu

Department of Mechanical Engineering, National University of Singapore

<hr style="border: 2px solid gray;"></hr>

This repository contains the code to establish the two-level hierarchical RL framework for the autonomous chemotactic navigation of the microrobots. Here, we focus on a flagellar swimmer with $9$ hinges and an ameboid swimmer with $20$ hinges.

### Environment
```
pip install virtualenv
virtualenv -p python3.7.17 myenv
source myenv/bin/activate
workon myenv
pip install -r requirements.txt
```

### Supported cases
1. Low level: acquisition of primitive swimming policies. 
```
python discretization.py
python train.py
```

2. High level: chemotactic navigation towards a static chemical source, towards a moving chemical source, within an ambient flow, and through a narrow constriction. 
```
python discretization.py
#The following execuation is only for the constricted scenario.
python constriction_discrete.py
#
python train.py
```

### Citation
PLease consider citing our [paper](https://arxiv.org/pdf/2408.07346) if you find it useful:
```bibtex
@article{xiong2024enabling,
  title={Enabling microrobotic chemotaxis via reset-free hierarchical reinforcement learning},
  author={Xiong, Tongzhao and Liu, Zhaorong and Ong, Chong Jin and Zhu, Lailai},
  journal={arXiv preprint arXiv:2408.07346},
  year={2024}
}
```
