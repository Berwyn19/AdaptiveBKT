# Adaptive Problem Selection in Competitive Physics

This project implements an adaptive problem selection system for high school competitive physics. The system aims to dynamically recommend problems to students based on their performance history and statistical analysis of previous students’ performance.

## Key Features

- **Bayesian Knowledge Tracing (BKT)**: Uses probabilistic parameters to estimate students' knowledge and update it based on their performance.
- **Beta Distribution for Problem Difficulty**: Models problem difficulty using Beta distributions and updates based on historical student performance.
- **Adaptive Learning Rate**: Adjusts the learning rate based on student performance metrics to ensure appropriate problem difficulty.
- **Dynamic Problem Selection**: Chooses problems with difficulty levels that best match the student's current knowledge.
- **Outlier Detection**: Identifies and handles outliers in student performance data to maintain system accuracy.

## Datasets

- **Time Data**: Simulated times for solving problems, generated with normal distributions.
- **Difficulty Data**: Difficulty levels derived from student performance on problems.

## Approach

1. **Bayesian Knowledge Tracing**: Updates the probability of knowing how to solve a problem based on student performance.
2. **Beta Distribution**: Models and updates the probability of getting problems right or wrong.
3. **Adaptive Learning Rate**: Adjusts learning rates based on performance variance and mean.
4. **Problem Difficulty and Selection**: Evaluates and selects problems based on difficulty and student knowledge.
5. **Outlier Detection**: Handles anomalous data to maintain system reliability.

For more details on the implementation and mathematical models used, refer to the [detailed documentation](AdaptiveBKT.pdf).
