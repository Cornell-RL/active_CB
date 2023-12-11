# Contextual Bandit with Active Learning

This is an implementation of the preference-based active learning algorithm for contextual bandit outlined in [Contextual Bandits and Imitation Learning via Preference-Based Active Queries](https://arxiv.org/abs/2307.12926). This paper considers the problem of contextual bandits and imitation learning, where the learner lacks direct knowledge of the executed action's reward. Under the assumption that the learner has access to a function class that can represent the expert's preference model under appropriate link functions, the paper proposed an algorithm that leverages an online regression oracle with respect to this function class for choosing its actions and deciding when to query.

## Installation
```bash
git clone https://github.com/Cornell-RL/active_CB.git
cd active_cb
pip install numpy pandas torch ucimlrepo
```

## Usage
```bash
python algo.py
```
This will run the preference learning algorithm on the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris). To run the reward learning algorithm or start with a different dataset, please follow
```bash
usage: algo.py [-h] [--dataset DATASET] [--query QUERY] [--model MODEL]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Name of the dataset (iris/car/knowledge)
  --query QUERY      Query type (active/passive)
  --model MODEL      Model type (reward/preference)
```
Running 1000 training iterations on the Iris dataset takes roughly three hours with evaluation. It is expected that running the algorithm on multi-class classification datasets with a large number of classes will take more episodes to converge and will take require a longer runtime.

## Results
Here are the results on the Iris, Car Evaluation, and User Knowledge Modeling datasets. The hyperparameters required by the algorithm are set in the training loop based on the dataset.

![Iris](https://github.com/Cornell-RL/active_CB/assets/59858888/88e5beb2-845d-4257-9427-d5f11e7163ba)

![Car](https://github.com/Cornell-RL/active_CB/assets/59858888/87e2886b-407c-43d1-9ef7-dee848b7e7e4)

![User Knowledge](https://github.com/Cornell-RL/active_CB/assets/59858888/e352d0a6-0c64-41d7-9787-f0676e46d0f7)

## Citation
```bash
@misc{sekhari2023contextual,
      title={Contextual Bandits and Imitation Learning via Preference-Based Active Queries}, 
      author={Ayush Sekhari and Karthik Sridharan and Wen Sun and Runzhe Wu},
      year={2023},
      eprint={2307.12926},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



