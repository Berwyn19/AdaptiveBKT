import numpy as np
import time
import os
import json
from scipy.stats import norm

NUM_ITERATION = 6

class Student:
    def __init__(self):
        # Initialize BKT parameters with Beta distribution parameters (α, β)
        self.p_G_alpha, self.p_G_beta = 1, 1
        self.p_S_alpha, self.p_S_beta = 1, 1

        # Learning rate for adaptive update
        self.default_lr = 2

        # Initial probabilities (assuming uniform prior)
        self.p_G = 0.5
        self.p_S = 0.5
        self.p_T = 0.5
        self.p_L_new = 0.5

        # List of performance history
        self.performance_history = []

        # Handling the JSON file
        self.time_data_file = '../database/time_data.json'
        self.time_data = self.load_time_data()

        # Ensure no repetition
        self.attempted_problems = []

    def load_time_data(self):
        if os.path.exists(self.time_data_file):
            with open(self.time_data_file, 'r') as file:
                return json.load(file)
        else:
            return {}

    def save_time_data(self):
        with open(self.time_data_file, 'w') as file:
            json.dump(self.time_data, file, indent=4)

    def update_time(self, time, problem_id):
        times = self.time_data[problem_id]
        if not self.is_outliers(problem_id, time):
            self.time_data[problem_id].append(time)
            self.save_time_data()

        # Calculate the average
        mu = np.mean(times)
        sigma = np.std(times)
        z_score = (time - mu)/sigma
        p_time = norm.cdf(z_score)

        prior_t = self.p_T
        prior_t = 0.2 * prior_t + 0.8 * (1 - p_time)
        self.p_T = prior_t
        print("p_T: {0}".format(self.p_T))

    def is_outliers(self, problem_id, time):
        # Calculate mean and standard deviation in time
        times = self.time_data[problem_id]
        mu = np.mean(times)
        sigma = np.std(times)

        # Calculate lower bound and upper bound for the definition of outlier
        lower_bound = mu - 3 * sigma
        upper_bound = mu + 3 * sigma

        if time < lower_bound or time > upper_bound:
            return True
        else:
            return False

    # Sample 20 times for the distribution and get the average of the probabilities
    def sample_beta(self, a, b, num_samples):
        return np.random.beta(a, b, num_samples).mean()

    # Update beta parameters for all probabilities P_G, P_S, and P_T, concept_performance is a list of indicator
    # variables in this case of size 3 which is 3 concepts incorporated in the problem
    def update_parameters(self, concept_performance, correct, problem_id, time):
        num_correct = 0
        # Count the number of correct concept questions
        print(concept_performance)
        for indicator in concept_performance:
            if indicator == 1:
                num_correct += 1
        num_wrong = len(concept_performance) - num_correct

        # Update the parameters of the Beta variables
        if correct == 1:
            self.p_G_alpha += num_wrong
            self.p_G_beta += num_correct
        else:
            self.p_S_alpha += num_correct
            self.p_S_beta += num_wrong

        # Update the probability p_L new by sampling for p_G and p_S
        p_g = self.sample_beta(self.p_G_alpha, self.p_G_beta, 40)
        p_s = self.sample_beta(self.p_S_alpha, self.p_S_beta, 40)

        print("p_g: {0}".format(p_g))
        print("p_s: {0}".format(p_s))
        print("p_L: {0}".format(self.p_L_new))

        if correct == 1:
            p_l = self.p_L_new * (1 - p_s) / (self.p_L_new * (1 - p_s) + (1 - self.p_L_new) * p_g)
            print("pl: {0}".format(p_l))
        else:
            p_l = self.p_L_new * p_s / (self.p_L_new * p_s + (1 - self.p_L_new) * (1 - p_g))
            print("pl: {0}".format(p_l))

        # Incorporating the adaptive learning rate
        adaptive_lr = self.calculate_adaptive_lr(self.performance_history)

        self.p_L_new = self.p_L_new + (p_l - self.p_L_new) * adaptive_lr
        print("milestone1: {0}".format(self.p_L_new))

        # Update the time factor
        self.update_time(time, problem_id)

        # Final update involving p_T, the time factor
        self.p_L_new = 0.5 * self.p_L_new + 0.5 * self.p_T
        print("milestone2: {0}".format(self.p_L_new))

    # This learning rate will be used to prevent p_L_new from experiencing significant jump
    # by calculating the mean and variance of the performance history
    def calculate_adaptive_lr(self, performance_history):
        if len(performance_history) < 2:
            print("aku disini")
            return self.default_lr

        # Calculating mean and variance of the performance history list
        performance_mean = np.mean(self.performance_history)
        performance_var = np.var(self.performance_history)

        # Calculating the adaptive lr, if mean is low, learning rate should be low because
        # that means the student hasn't mastered the concept. If variance is high,
        # the learning rate should also be low because that shows inconsistency in
        # performance
        guess_probability = self.sample_beta(self.p_G_alpha, self.p_G_beta, 40)
        adaptive_lr = self.default_lr * performance_mean / (1 + 5 * performance_var + 3 * guess_probability)
        print("mean: {0}".format(performance_mean))
        print("var: {0}".format(performance_var))
        print("lr: {0}".format(adaptive_lr))
        if adaptive_lr < 1.5:
            adaptive_lr += 1.5

        return adaptive_lr

    def determine_next_problem(self):
        scores = [abs(self.p_L_new - (1 - self.sample_beta(problems[str(i + 1)]["a"], problems[str(i + 1)]["b"], 1000)))
                  for i in range(len(problems))]

        # Sort problems by their scores but maintain their indices
        sorted_problems = sorted(enumerate(scores), key=lambda x: x[1])

        for index, score in sorted_problems:
            problem_id = str(index + 1)
            if problem_id not in self.attempted_problems:
                return problem_id

        return None

def update_difficulty(problem_id, is_correct):
    with open('../database/problems.json', 'r') as file:
        current_state = json.load(file)

    if is_correct:
        current_state[problem_id]["a"] += 1
    else:
        current_state[problem_id]["b"] += 1
        
    with open('../database/problems.json', 'w') as file:
        json.dump(current_state, file, indent=4)

# Simulate problem solving
def simulate_problem_solving(student, problem_id):
    with open('../database/concept_problems.json', 'r') as file:
        concept_questions = json.load(file)

    with open('../database/problems.json', 'r') as file:
        problems = json.load(file)

    # Start timer
    start_time = time.time()
    print("start {0}".format(start_time))
    print(student.p_L_new)

    # Ask student the conceptual questions
    print("Problem {0}".format(problem_id))
    print(concept_questions[problem_id]["concept1"])
    print("1. {0}".format(concept_questions[problem_id]["choice1"]))
    print("2. {0}".format(concept_questions[problem_id]["choice2"]))
    print("3. {0}".format(concept_questions[problem_id]["choice3"]))
    concept1_ans = input("Choose the correct number for concept 1: ")

    print(concept_questions[problem_id]["concept2"])
    print("1. {0}".format(concept_questions[problem_id]["choice4"]))
    print("2. {0}".format(concept_questions[problem_id]["choice5"]))
    print("3. {0}".format(concept_questions[problem_id]["choice6"]))
    concept2_ans = input("Choose the correct number for concept 2: ")

    print(concept_questions[problem_id]["concept3"])
    print("1. {0}".format(concept_questions[problem_id]["choice7"]))
    print("2. {0}".format(concept_questions[problem_id]["choice8"]))
    print("3. {0}".format(concept_questions[problem_id]["choice9"]))
    concept3_ans = input("Choose the correct number for concept 3: ")

    print(problems[problem_id]["problem_statement"])
    print("1. {0}".format(problems[problem_id]["choice1"]))
    print("2. {0}".format(problems[problem_id]["choice2"]))
    print("3. {0}".format(problems[problem_id]["choice3"]))
    ans = input("Choose the correct number: ")

    end_time = time.time()
    print("end {0}".format(end_time))
    time_required = end_time - start_time
    print("delta {0}".format(time_required))

    ans_list = []

    # Checking conceptual answers
    if concept1_ans == concept_questions[problem_id]["ans1"]:
        ans_list.append(1)
    elif concept1_ans != concept_questions[problem_id]["ans1"]:
        ans_list.append(0)
    if concept2_ans == concept_questions[problem_id]["ans2"]:
        ans_list.append(1)
    elif concept2_ans != concept_questions[problem_id]["ans2"]:
        ans_list.append(0)
    if concept3_ans == concept_questions[problem_id]["ans3"]:
        ans_list.append(1)
    elif concept3_ans != concept_questions[problem_id]["ans3"]:
        ans_list.append(0)

    # Checking the answer
    if ans == problems[problem_id]["ans"]:
        correct = 1
        if not student.is_outliers(problem_id, time_required):
            problems[problem_id]["a"] += 1
            update_difficulty(problem_id, True)
    else:
        correct = 0
        update_difficulty(problem_id, False)

    # Score for performance history is calculated via weighted average between
    # the concept performance and the performance for the actual question, where
    # the latter was weighed heavier

    w1 = 0.3
    w2 = 0.7
    student.performance_history.append(w1 * sum(ans_list)/3 + w2 * correct)
    student.attempted_problems.append(problem_id)

    # Number of students who have attempted the question
    n_attempts = problems[problem_id]["a"] + problems[problem_id]["b"] - 2

    # Update the parameters
    student.update_parameters(ans_list, correct, problem_id, time_required)


def simulate_real_time(student):
    # Always start with problem 1
    simulate_problem_solving(student, "1")
    next_problem = student.determine_next_problem()
    for i in range(NUM_ITERATION):
        simulate_problem_solving(student, str(next_problem))
        next_problem = student.determine_next_problem()


if __name__ == "__main__":
    # Initiating an instance of the Student class
    test_student = Student()

    # Simulate the test
    simulate_real_time(test_student)