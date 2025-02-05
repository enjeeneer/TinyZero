import re

import numpy as np


def extract_prediction(solution_str) -> float:
    """Extract the forecast/prediction from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    match = re.search(r"'probability':\s*([\d.]+)", solution_str)
    if match:
        prediction = float(match.group(1))
    else:
        prediction = None
        print("No prediction found.")

    return prediction

def compute_score(solution_str, ground_truth: bool, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: bool (event happened or didn't happen)
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    
    prediction = extract_prediction(solution_str=solution_str)

    # get log score
    if prediction is not None:
        if ground_truth:
            reward = np.log(prediction)
        else:
            reward = np.log(1 - prediction)

    # arbitrarily low reward for no prediction (i.e. bad formatting)
    else:
        reward = -10

    print(f"--------------------------------")
    print(f"Target: {int(ground_truth)} | Prediction: {prediction}")
    print(f"Reward (log score): {reward}")

    return reward

