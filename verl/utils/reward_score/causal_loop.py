"""
Reward scoring for Causal Loop Prediction Task.

This module evaluates model responses for causal reasoning problems,
computing rewards based on:
1. Correct final state prediction (full score)
2. Partial correctness (partial score based on correct variables)
3. Valid format but wrong answer (format score)
4. Invalid format (zero score)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

# Set up logger for debug output
logger = logging.getLogger(__name__)


def extract_answer(solution_str: str) -> Optional[str]:
    """Extract the answer from the solution string.

    Looks for content within <answer>...</answer> tags.

    Args:
        solution_str: Full model output string

    Returns:
        Extracted answer string or None if not found
    """
    # Remove everything before the assistant response
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # Find all answer tags and take the last one
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        return matches[-1].group(1).strip()
    return None


def parse_state(answer_str: str, expected_vars: List[str]) -> Optional[Dict[str, int]]:
    """Parse a state string into a dictionary of variable values.

    Expected formats:
    - "A=1, B=2, C=3"
    - "A = 1, B = 2, C = 3"
    - "A: 1, B: 2, C: 3"
    - "A=1 B=2 C=3" (space separated)

    Args:
        answer_str: The answer string to parse
        expected_vars: List of expected variable names

    Returns:
        Dictionary mapping variable names to values, or None if parsing fails
    """
    if not answer_str:
        return None

    state = {}

    # Try different parsing patterns
    patterns = [
        r'([A-Z])\s*[=:]\s*(-?\d+)',  # A=1, A: 1, A = 1
    ]

    for pattern in patterns:
        matches = re.findall(pattern, answer_str)
        if matches:
            for var, val in matches:
                if var in expected_vars:
                    try:
                        state[var] = int(val)
                    except ValueError:
                        continue

    # Check if we found all expected variables
    if set(state.keys()) == set(expected_vars):
        return state

    # If not all found, try more aggressive parsing
    # Look for any integer after each variable letter
    for var in expected_vars:
        if var not in state:
            # Find patterns like "A is 5" or "A becomes 5" or just "A 5"
            patterns = [
                rf'{var}\s*[=:]\s*(-?\d+)',
                rf'{var}\s+(?:is|becomes|equals?)\s+(-?\d+)',
                rf'{var}\s+(-?\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, answer_str)
                if match:
                    try:
                        state[var] = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

    if state:
        return state
    return None


def compute_score(
    solution_str: str,
    ground_truth: Dict,
    method: str = 'strict',
    format_score: float = 0.1,
    partial_base: float = 0.3,
    score: float = 1.0,
) -> float:
    """Compute the reward score for a causal loop prediction.

    Scoring tiers:
    1.0 - All variables correct
    0.3-0.9 - Partial correctness (proportional to correct variables)
    0.1 - Valid format but wrong values
    0.0 - Invalid format or no answer

    Args:
        solution_str: The full model output
        ground_truth: Dictionary containing:
            - final_state: Dict[str, int] mapping variable names to values
            - variables: List[str] of variable names
            - level: int difficulty level (1-4)
            - num_steps: int number of simulation steps
        method: Scoring method ('strict' for exact match only)
        format_score: Score for valid format but wrong answer
        partial_base: Base score for partial correctness
        score: Score for fully correct answer

    Returns:
        Float reward score
    """
    final_state = ground_truth['final_state']
    variables = ground_truth['variables']
    level = ground_truth.get('level', 1)
    num_steps = ground_truth.get('num_steps', 1)

    # Extract answer from response
    answer = extract_answer(solution_str)

    # Log debug info
    logger.debug(f"Level: {level} | Steps: {num_steps}")
    logger.debug(f"Expected: {final_state}")
    logger.debug(f"Extracted answer: {answer}")

    if answer is None:
        logger.debug(f"Score: 0 (no answer found)")
        logger.debug(f"Full solution: {solution_str[:500]}...")
        return 0.0

    # Parse the answer into a state dictionary
    parsed_state = parse_state(answer, variables)

    if parsed_state is None:
        logger.debug(f"Score: 0 (could not parse state)")
        return 0.0

    # Count correct variables
    correct_count = 0
    total_count = len(variables)

    for var in variables:
        expected = final_state.get(var)
        predicted = parsed_state.get(var)
        if expected is not None and predicted is not None and expected == predicted:
            correct_count += 1

    logger.debug(f"Parsed state: {parsed_state}")
    logger.debug(f"Correct: {correct_count}/{total_count}")

    # Compute score
    if correct_count == total_count:
        # Full score for all correct
        final_score = score
        logger.debug(f"Score: {final_score} (all correct)")
    elif correct_count > 0:
        # Partial score based on correctness ratio
        ratio = correct_count / total_count
        final_score = partial_base + (score - partial_base) * ratio * 0.9
        logger.debug(f"Score: {final_score:.3f} (partial: {correct_count}/{total_count})")
    else:
        # Format is valid but all values wrong
        final_score = format_score
        logger.debug(f"Score: {final_score} (format ok, all wrong)")

    return final_score


def compute_score_batch(
    solutions: List[str],
    ground_truths: List[Dict],
    **kwargs,
) -> List[float]:
    """Compute scores for a batch of solutions.

    Args:
        solutions: List of model outputs
        ground_truths: List of ground truth dictionaries

    Returns:
        List of reward scores
    """
    return [
        compute_score(sol, gt, **kwargs)
        for sol, gt in zip(solutions, ground_truths)
    ]


def analyze_errors(
    solutions: List[str],
    ground_truths: List[Dict],
) -> Dict:
    """Analyze error patterns in model predictions.

    Returns statistics about:
    - Format errors (couldn't parse)
    - Systematic errors (always wrong on certain variable)
    - Off-by-one errors
    - Level-wise accuracy

    Args:
        solutions: List of model outputs
        ground_truths: List of ground truth dictionaries

    Returns:
        Dictionary with error analysis statistics
    """
    stats = {
        'total': len(solutions),
        'format_errors': 0,
        'no_answer': 0,
        'fully_correct': 0,
        'partial_correct': 0,
        'all_wrong': 0,
        'by_level': {1: [], 2: [], 3: [], 4: []},
        'off_by_one': 0,
        'variable_accuracy': {},
    }

    for sol, gt in zip(solutions, ground_truths):
        final_state = gt['final_state']
        variables = gt['variables']
        level = gt.get('level', 1)

        answer = extract_answer(sol)
        if answer is None:
            stats['no_answer'] += 1
            stats['by_level'][level].append(0)
            continue

        parsed = parse_state(answer, variables)
        if parsed is None:
            stats['format_errors'] += 1
            stats['by_level'][level].append(0)
            continue

        correct = 0
        for var in variables:
            expected = final_state.get(var, 0)
            predicted = parsed.get(var, None)

            if var not in stats['variable_accuracy']:
                stats['variable_accuracy'][var] = {'correct': 0, 'total': 0, 'off_by_one': 0}

            stats['variable_accuracy'][var]['total'] += 1

            if predicted == expected:
                correct += 1
                stats['variable_accuracy'][var]['correct'] += 1
            elif predicted is not None and abs(predicted - expected) == 1:
                stats['off_by_one'] += 1
                stats['variable_accuracy'][var]['off_by_one'] += 1

        if correct == len(variables):
            stats['fully_correct'] += 1
            stats['by_level'][level].append(1)
        elif correct > 0:
            stats['partial_correct'] += 1
            stats['by_level'][level].append(correct / len(variables))
        else:
            stats['all_wrong'] += 1
            stats['by_level'][level].append(0)

    # Compute level-wise averages
    for level in stats['by_level']:
        scores = stats['by_level'][level]
        if scores:
            stats['by_level'][level] = {
                'count': len(scores),
                'mean': sum(scores) / len(scores),
                'full_correct_rate': sum(1 for s in scores if s == 1) / len(scores),
            }

    # Compute variable accuracy rates
    for var in stats['variable_accuracy']:
        va = stats['variable_accuracy'][var]
        if va['total'] > 0:
            va['accuracy'] = va['correct'] / va['total']
            va['off_by_one_rate'] = va['off_by_one'] / va['total']

    return stats
