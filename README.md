# Introduction to Artificial Intelligence – Laboratory Exercises (Python)

This repository contains my Python solutions for all laboratory assignments in the Introduction to Artificial Intelligence course (academic year 2022/23). Each exercise lives in its own folder and can be run independently. The tasks cover fundamental AI topics: state space search, automated reasoning, decision trees, and evolutionary optimization of neural networks. The goal was to implement everything from scratch (Python standard library only, optional numpy in Exercise 4) following the given autograder interfaces.

## Exercise 1: State Space Search
Implements:
- Data loading for a generic graph/state space and heuristic files
- Search algorithms: Breadth-First Search (BFS), Uniform Cost Search (UCS), A* (tie-breaking alphabetically)
- Heuristic evaluation utilities: optimism check (h ≤ h*), consistency check (triangle inequality)

Command-line usage examples:
`python solution.py --alg bfs --ss path/to/state_space.txt`
`python solution.py --alg astar --ss path/to/state_space.txt --h path/to/heuristic.txt`

Add flags `--check-optimistic` or `--check-consistent` for heuristic validation.

Outputs follow the specification (`FOUND_SOLUTION`, `STATES_VISITED`, etc.).

## Exercise 2: Automated Reasoning and Cooking Assistant
**Part 1** – Refutation resolution with:
- Set-of-support strategy
- Deletion of redundant/irrelevant clauses
- Derivation trace up to NIL and final `[CONCLUSION]`

**Part 2** – Cooking assistant:
- Dynamic knowledge base (additions `+`, deletions `-`, queries `?`)
- Uses resolution internally to answer queries

Run:
`python solution.py resolution path/to/clauses.txt`
`python solution.py cooking path/to/clauses.txt path/to/commands.txt`

## Exercise 3: Decision Tree (ID3)
**Features:**
- CSV dataset loading (last column = class label)
- ID3 training with information gain (entropy base 2)
- Branch printing: `[BRANCHES]` with `depth:feature=value ... leaf_label`
- Prediction on test set: `[PREDICTIONS]`
- Evaluation: accuracy (5 decimals) and confusion matrix (labels sorted alphabetically; only labels present in true/predicted)
- Optional maximum depth (third CLI argument). Depth 0 collapses to a single majority-class leaf with alphabetical tie-break.

Run:
`python solution.py path/to/train.csv path/to/test.csv [max_depth]`

## Exercise 4: Genetic Algorithm Optimized Neural Networks
Implements feedforward neural networks (architectures: 5s, 20s, 5s5s) and a generational genetic algorithm:
- Network: sigmoid hidden layers, linear output; weights initialized N(0, 0.01)
- GA components: fitness-proportional selection, arithmetic mean crossover, Gaussian mutation (std=K, per-weight probability p), elitism
- Periodic training error print every 2000 generations and final test error

Run:
`python solution.py --train train.csv --test test.csv --nn 5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000`
(Optionally uses numpy for numerical operations.)

## Requirements
- Python 3.7+ (as per assignment environment)
- No external libraries except optional numpy in Exercise 4

## Notes
All implementations conform to the autograder input/output specifications. The code avoids external dependencies, emphasizes clarity, and demonstrates core AI algorithms end-to-end.
