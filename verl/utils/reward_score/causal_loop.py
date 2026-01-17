"""
Reward scoring for Causal Loop Prediction Task.

This module evaluates model responses for causal reasoning problems,
computing rewards based on:
1. Correct final state prediction (full score)
2. Partial correctness (partial score based on correct variables)
3. Valid format but wrong answer (format score)
4. Invalid format (zero score)

NEW: Trajectory validation support
- Validates intermediate reasoning steps when trajectory is provided
- Penalizes correct final answers reached through wrong intermediate steps
- Encourages genuine sequential reasoning vs lucky guessing
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

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


def extract_intermediate_states(solution_str: str, expected_vars: List[str]) -> List[Dict[str, int]]:
    """Extract intermediate states from model's reasoning.

    Looks for state descriptions in the <think> section that show
    intermediate steps of computation.

    Expected patterns in thinking:
    - "After rule 1: A=5, B=3"
    - "Step 1: A=5, B=3, C=2"
    - "State after applying rule 1: A=5"
    - Just variable assignments like "A becomes 5" or "A is now 5"

    Args:
        solution_str: Full model output string
        expected_vars: List of expected variable names

    Returns:
        List of parsed state dictionaries in order of appearance
    """
    # Extract the thinking section
    think_match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    if not think_match:
        return []

    think_content = think_match.group(1)
    intermediates = []

    # Find all lines that look like state descriptions
    # Pattern 1: "After rule X: A=5, B=3" or "Step X: A=5, B=3"
    step_patterns = [
        r'(?:after|step|state|applying)[^:]*:\s*([A-Z\s=,\d]+)',
        r'(?:now|becomes|results?)[^:]*:\s*([A-Z\s=,\d]+)',
    ]

    # Split into lines and process
    lines = think_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse state from each line
        state = {}
        for var in expected_vars:
            # Look for patterns like "A=5", "A = 5", "A is 5", "A becomes 5"
            patterns = [
                rf'{var}\s*[=:]\s*(-?\d+)',
                rf'{var}\s+(?:is|becomes|equals?|now)\s+(-?\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        state[var] = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

        # Only add if we found at least one variable
        if state:
            intermediates.append(state)

    return intermediates


def validate_trajectory(
    solution_str: str,
    expected_trajectory: List[Dict[str, int]],
    expected_vars: List[str],
) -> float:
    """Validate model's intermediate reasoning against expected trajectory.

    Compares the intermediate states extracted from the model's thinking
    against the expected trajectory of states.

    Args:
        solution_str: Full model output string
        expected_trajectory: List of expected intermediate states
        expected_vars: List of expected variable names

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if not expected_trajectory:
        return 1.0  # No trajectory to validate

    # Extract model's intermediate states
    model_intermediates = extract_intermediate_states(solution_str, expected_vars)

    if not model_intermediates:
        # Model didn't show intermediate states - can't validate
        # Return neutral score (don't penalize or reward)
        return 0.5

    # Compare trajectories
    # We use a flexible matching - check if model states appear in expected order
    total_comparisons = 0
    correct_comparisons = 0

    # For each expected intermediate state, check if model has a similar state
    for i, expected_state in enumerate(expected_trajectory):
        # Find the closest matching model state (if any)
        best_match_score = 0.0

        for model_state in model_intermediates:
            # Count matching variables
            matches = 0
            total_vars = 0

            for var in expected_vars:
                if var in expected_state:
                    total_vars += 1
                    if var in model_state and model_state[var] == expected_state[var]:
                        matches += 1

            if total_vars > 0:
                match_score = matches / total_vars
                best_match_score = max(best_match_score, match_score)

        total_comparisons += 1
        correct_comparisons += best_match_score

    if total_comparisons == 0:
        return 1.0

    return correct_comparisons / total_comparisons


def compute_score(
    solution_str: str,
    ground_truth: Dict,
    method: str = 'strict',
    format_score: float = 0.1,
    partial_base: float = 0.3,
    score: float = 1.0,
    validate_intermediates: bool = False,
    intermediate_penalty: float = 0.2,
) -> float:
    """Compute the reward score for a causal loop prediction.

    Scoring tiers:
    1.0 - All variables correct
    0.3-0.9 - Partial correctness (proportional to correct variables)
    0.1 - Valid format but wrong values
    0.0 - Invalid format or no answer

    With trajectory validation (validate_intermediates=True):
    - Final score is adjusted based on intermediate reasoning accuracy
    - Encourages correct reasoning process, not just correct answers

    Args:
        solution_str: The full model output
        ground_truth: Dictionary containing:
            - final_state: Dict[str, int] mapping variable names to values
            - variables: List[str] of variable names
            - level: int difficulty level (1-4)
            - num_steps: int number of simulation steps
            - trajectory: Optional[List[Dict]] intermediate states for validation
        method: Scoring method ('strict' for exact match only)
        format_score: Score for valid format but wrong answer
        partial_base: Base score for partial correctness
        score: Score for fully correct answer
        validate_intermediates: Whether to validate intermediate reasoning steps
        intermediate_penalty: How much to penalize wrong intermediate reasoning (0-1)

    Returns:
        Float reward score
    """
    final_state = ground_truth['final_state']
    variables = ground_truth['variables']
    level = ground_truth.get('level', 1)
    num_steps = ground_truth.get('num_steps', 1)
    trajectory = ground_truth.get('trajectory', None)

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

    # Compute base score
    if correct_count == total_count:
        # Full score for all correct
        final_score = score
        logger.debug(f"Base score: {final_score} (all correct)")
    elif correct_count > 0:
        # Partial score based on correctness ratio
        ratio = correct_count / total_count
        final_score = partial_base + (score - partial_base) * ratio * 0.9
        logger.debug(f"Base score: {final_score:.3f} (partial: {correct_count}/{total_count})")
    else:
        # Format is valid but all values wrong
        final_score = format_score
        logger.debug(f"Base score: {final_score} (format ok, all wrong)")

    # NEW: Apply trajectory validation if enabled and trajectory is available
    if validate_intermediates and trajectory is not None and len(trajectory) > 0:
        intermediate_accuracy = validate_trajectory(solution_str, trajectory, variables)
        logger.debug(f"Intermediate accuracy: {intermediate_accuracy:.3f}")

        # Adjust final score based on intermediate reasoning quality
        # If intermediates are wrong but answer is right, reduce reward
        # This penalizes "lucky guessing" and rewards genuine reasoning
        if intermediate_accuracy < 1.0:
            penalty = intermediate_penalty * (1.0 - intermediate_accuracy)
            final_score = final_score * (1.0 - penalty)
            logger.debug(f"Score after trajectory penalty: {final_score:.3f}")

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
