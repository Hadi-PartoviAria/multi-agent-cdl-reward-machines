# Multi-Agent Causal Dynamics Learning for Temporally Extended Tasks with Reward Machine Inference

Code for the paper by Hadi Partovi Aria, Zhe Xu. American Control Conference (ACC), 2025.

Each agent works on a state abstraction from conditional mutual information (CMI) based causal inference, learns its own reward machine from traces and runs Q-learning over the abstract state and reward machine state. The code covers both decentralized and centralized setups.

## Layout

`src/run.py` is the entry point; the experiment is selected in `src/buttons_config.py` and `src/rendezvous_config.py`. `src/Environments/coop_buttons/` is the cooperative buttons task (first case study) and `src/Environments/rendezvous/` is the gridworld used for the two-agent dice stacking task. `src/Agent/` and `src/experiments/` have the decentralized (`dqprm.py`, `iql.py`) and centralized learners. Reward machine definitions are in `experiments/`.

## Running

```bash
pip install -r requirements.txt
cd src && python run.py
```

The CMI-based causal inference for the state abstraction is in https://github.com/Hadi-PartoviAria/causal-dynamics-reward-machines.

## Credits

This code builds on Neary et al., "Reward Machines for Cooperative Multi-Agent Reinforcement Learning" (https://github.com/cyrusneary/rm-cooperative-marl, MIT License, copyright 2021 Cyrus Neary). That license is kept in `LICENSE-rm-cooperative-marl`.

## Citation

```bibtex
  @inproceedings{aria2025macdl,
    title = {Multi-Agent Causal Dynamics Learning for Temporally Extended Tasks with Reward Machine Inference},
    author = {Hadi Partovi Aria and Zhe Xu},
    booktitle = {2025 American Control Conference (ACC)},
pages = {3628--3633},
    year = {2025}
  }
```

## License

MIT, see `LICENSE`.
