

# =========================================
# AI-Evolved Bin Packing Heuristics
# =========================================
#
# This script evolves and benchmarks heuristics for the classic bin packing problem
# using LLMs (OpenAI or local). It is inspired by:
#   - AlphaEvolve (Novikov et al., 2025)
#   - FunSearch (Romera-Paredes et al., 2024)
#
# Author: Mehdi Soleimanifar
#
# Requirements:
#   - Python 3.8+
#   - numpy, openai
#   - Dataset files: OR3_dataset.json, Weibull5k_dataset.json
#
# Usage:
#   python main__.py
# =========================================




import json
import numpy as np
import random



########## 1. Imports and Data Loading ##########

from helpers import (
    l1_bound, l1_bound_dataset,
    first_fit_heuristic, best_fit_heuristic,
    discovered_heuristic_or, discovered_heuristic_weibull, funsearch_heuristic_weibull,
    is_valid_packing, get_valid_bin_indices
)



########## 2. LLM API Config ##########

USE_LOCAL_LLM = False  # Set to True to use a local LLM server via lmstudio
VERBOSE_LLM_CODE = True

if USE_LOCAL_LLM:
    API_CONFIG = {
        "provider": "local",
        "api_key": "sk-local",  # Not used, but required by openai lib
        "model": "lmstudio",
        "base_url": "http://localhost:1234/v1"
    }
    import openai
    openai_client = openai.OpenAI(
        api_key=API_CONFIG["api_key"],
        base_url=API_CONFIG["base_url"]
    )
else:
    API_CONFIG = {
        "provider": "openai",
        "api_key": "sk-...", # Replace with your OpenAI API key
        "model": "gpt-4.1-nano"
    }
    import openai
    openai_client = openai.OpenAI(api_key=API_CONFIG["api_key"])




# --- Load datasets ---
with open('OR3_dataset.json') as f:
    or3 = json.load(f)
with open('Weibull5k_dataset.json') as f:
    weibull5k = json.load(f)

datasets = {
    'OR3': or3,
    'Weibull 5k': weibull5k
}




########## 3. Core Bin Packing Logic & LLM Evolution ##########

# --- Online bin packing (fixed bin array) ---
def online_binpack2(items, bins, heuristic_fn):
    """Pack items into bins using a heuristic function."""
    packing = [[] for _ in bins]
    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        if len(valid_bin_indices) == 0:
            continue
        priorities = heuristic_fn(item, bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins

# --- Evaluate a heuristic on a dataset ---
def evaluate2(instances, heuristic_fn):
    """Evaluate a heuristic function on a set of binpacking instances."""
    num_bins = []
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        bins = np.array([capacity for _ in range(len(items))])
        packing, _ = online_binpack2(items, bins, heuristic_fn)
        num_bins.append(len(packing))
    return -np.mean(num_bins)

# --- LLM Heuristic class ---
class LLMHeuristic:
    def __init__(self, code: str):
        self.code = code

    def __call__(self, item, bins):
        import numpy as np
        local_scope = {"item": item, "bins": bins, "scores": None}
        global_scope = {"np": np}
        raw = self.code.strip().splitlines()
        if raw and raw[0].startswith("```"):
            raw = raw[1:]
        if raw and raw[-1].startswith("```"):
            raw = raw[:-1]
        clean_code = "\n".join(raw)
        try:
            exec(clean_code, global_scope, local_scope)
            scores = local_scope["scores"]
            scores = np.asarray(scores, dtype=float).flatten()
            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
            if scores.shape != (len(bins),):
                return np.zeros(len(bins)), f"LLM heuristic returned scores with invalid shape: {scores.shape}"
            return scores, None
        except Exception as e:
            # If np is not defined, try again with np injected (defensive, but should not be needed)
            if "name 'np' is not defined" in str(e):
                try:
                    local_scope["np"] = np
                    exec(clean_code, {}, local_scope)
                    scores = local_scope["scores"]
                    scores = np.asarray(scores, dtype=float).flatten()
                    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
                    if scores.shape != (len(bins),):
                        return np.zeros(len(bins)), f"LLM heuristic returned scores with invalid shape: {scores.shape}"
                    return scores, None
                except Exception as e2:
                    return np.zeros(len(bins)), str(e2)
            return np.zeros(len(bins)), str(e)


# --- Extract strategy comment ---
def extract_strategy_comment(code: str) -> str:
    """
    Extracts the first comment block (strategy description) from the code.
    Assumes the strategy comment is at the top and starts with '#'.
    """
    lines = code.strip().splitlines()
    comment_lines = []
    for line in lines:
        if line.strip().startswith("#"):
            comment_lines.append(line.strip())
        elif comment_lines:
            break  # Stop at first non-comment after comments
    return " ".join(comment_lines) if comment_lines else ""


# --- LLM Heuristic Generation ---
def llm_generate_binpack_heuristic(
    existing_code: str,
    performance: float,
    api_config: dict,
    error_message: str = None,
    openai_client=None,
    mutation_type: str = "refine",
    strategy_history: list = None
):
    # --- Common system prompt ---
    system_prompt = (
        "You are an expert in combinatorial optimization. "
        "Your task is to write a Python function body that, given an item and a numpy array of bin capacities, "
        "returns a numpy array of scores (higher is better) for each bin. "
        "The function signature is: def heuristic(item, bins): ... return scores "
        "You may use numpy as np. "
        "Provide ONLY the function body, assigning the result to the variable 'scores'. "
        "IMPORTANT: The variable 'scores' MUST be a 1D numpy array of floats, shape (len(bins),), with no nested arrays or lists. "
    )

    # --- Prepare strategy history for invent ---
    history_str = ""
    if strategy_history and len(strategy_history) > 0:
        history_str = "\n".join(
            [f"Strategy {i+1}: {comment}" for i, comment in enumerate(strategy_history)]
        )

    # --- Refine prompt ---
    if mutation_type == "refine":
        perf_str = f"{performance:.2f}%" if performance is not None else "N/A"
        user_prompt = (
            f"The current heuristic has an average excess bins of {perf_str}:\n"
            f"```python\n{existing_code}\n```\n"
            "Please come up with a significantly modified code that improves the heuristic. You may for example:\n"
            "- Change coefficients or parameters in a meaningful way\n"
            "- Modify the logic or mathematical expressions\n"
            "- Add or remove features\n"
            "- Try a different way to combine signals\n"
            "- Or make any other improvement you see fit\n"
            "IMPORTANTLY: The code MUST start with a sentence, commented description of the new strategy."
        )
        if error_message:
            user_prompt += f"\n\nThe previous code failed with this error:\n{error_message}\nPlease fix the error and try again."
    
    # --- Invent prompt ---
    else:  # mutation_type == "invent"
        user_prompt = (
            "Here is a summary of the strategies used so far:\n"
            f"{history_str}\n"
            "Invent a new heuristic for online bin packing that is qualitatively different from the above strategies. "
            " Your new algorithm should be grounded in reasoning about the problem and should not be a minor tweak of existing heuristics. "
            "You may use get inspired by any mathematical or algorithmic approach you see fit, such as BUT NOT limited to:"
            "- Double hashing\n"
            "- Cuckoo hashing\n"
            "- Hopscotch hashing\n"
            "- Or any other creative probing method\n"
            "IMPORTANTLY: The code MUST start with a sentence, commented description of the NEW strategy."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    resp = openai_client.chat.completions.create(
        model=api_config["model"],
        messages=messages,
        n=1,
    )
    code = resp.choices[0].message.content.strip()

    return code

# --- Save all LLM codes and scores with detailed reporting ---
def save_llm_history(llm_code_history, best_code, best_score, datasets, l1_bound_dataset,
                     first_fit_heuristic, best_fit_heuristic, discovered_heuristic_or,
                     discovered_heuristic_weibull, funsearch_heuristic_weibull, LLMHeuristic):
    with open("llm_binpack_heuristics.txt", "w") as f:
        f.write("=== All LLM-Generated Bin Packing Heuristics ===\n\n")
        for entry in llm_code_history:
            f.write(
                f"--- Gen {entry['generation']} | Round {entry['round']} | "
                f"Score: {entry['score']:.2f}%{' | BEST' if entry['is_best'] else ''} ---\n"
            )
            f.write(f"Parent: {entry['parent']}\n")
            f.write(f"Mutation: {entry['mutation_type']}\n")
            f.write(f"Tournament: {entry['tournament']}\n")
            f.write(f"Elite group: {entry['elite_group']}\n")
            f.write(entry['code'].strip() + "\n\n")
        f.write("=== Best LLM-Evolved Heuristic ===\n")
        f.write(f"Score: {best_score:.2f}%\n")

        # Also include the performance table and summary as printed
        f.write("\n=== Fraction of Excess Bins (Lower is Better) ===\n")
        f.write("NOTE: All evaluations use items in their original dataset order (no shuffling).\n")
        header = f"{'Heuristic':<20}{'OR3':>12}{'Weibull 5k':>15}\n"
        f.write(header)
        f.write('-' * len(header) + "\n")
        # Standard heuristics
        for heur_name, heur_fn in [
            ('First Fit', first_fit_heuristic),
            ('Best Fit', best_fit_heuristic)
        ]:
            row = f"{heur_name:<20}"
            for dataset_name in ['OR3', 'Weibull 5k']:
                dataset = datasets[dataset_name]
                num_bins_list = []
                l1_bounds = []
                for name in dataset:
                    instance = dataset[name]
                    items = instance['items']
                    capacity = instance['capacity']
                    bins = np.array([capacity for _ in range(len(items))])
                    packing, _ = online_binpack2(items, bins, heur_fn)
                    num_bins_list.append(len(packing))
                    l1_bounds.append(l1_bound(items, capacity))
                avg_bins = np.mean(num_bins_list)
                avg_l1 = np.mean(l1_bounds)
                excess = (avg_bins - avg_l1) / avg_l1 * 100
                row += f"{excess:>11.2f}%"
            f.write(row + "\n")

        # Discovered heuristic row (merged)
        row = f"{'Discovered':<20}"
        # OR3 with discovered_heuristic_or
        dataset = datasets['OR3']
        num_bins_list = []
        l1_bounds = []
        for name in dataset:
            instance = dataset[name]
            items = instance['items']
            capacity = instance['capacity']
            bins = np.array([capacity for _ in range(len(items))])
            packing, _ = online_binpack2(items, bins, discovered_heuristic_or)
            num_bins_list.append(len(packing))
            l1_bounds.append(l1_bound(items, capacity))
        avg_bins = np.mean(num_bins_list)
        avg_l1 = np.mean(l1_bounds)
        excess = (avg_bins - avg_l1) / avg_l1 * 100
        print(avg_bins, avg_l1, excess)
        row += f"{excess:>11.2f}%"
        # Weibull 5k with discovered_heuristic_weibull
        dataset = datasets['Weibull 5k']
        num_bins_list = []
        l1_bounds = []
        for name in dataset:
            instance = dataset[name]
            items = instance['items']
            capacity = instance['capacity']
            bins = np.array([capacity for _ in range(len(items))])
            packing, _ = online_binpack2(items, bins, discovered_heuristic_weibull)
            num_bins_list.append(len(packing))
            l1_bounds.append(l1_bound(items, capacity))
        avg_bins = np.mean(num_bins_list)
        avg_l1 = np.mean(l1_bounds)
        excess = (avg_bins - avg_l1) / avg_l1 * 100
        row += f"{excess:>14.2f}%"
        f.write(row + "\n")

        # FunSearch heuristic row
        row = f"{'FunSearch':<20}"
        for dataset_name, heur_fn in [
            ('OR3', funsearch_heuristic_weibull),
            ('Weibull 5k', funsearch_heuristic_weibull)
        ]:
            dataset = datasets[dataset_name]
            num_bins_list = []
            l1_bounds = []
            for name in dataset:
                instance = dataset[name]
                items = instance['items']
                capacity = instance['capacity']
                bins = np.array([capacity for _ in range(len(items))])
                packing, _ = online_binpack2(items, bins, heur_fn)
                num_bins_list.append(len(packing))
                l1_bounds.append(l1_bound(items, capacity))
            avg_bins = np.mean(num_bins_list)
            avg_l1 = np.mean(l1_bounds)
            excess = (avg_bins - avg_l1) / avg_l1 * 100
            row += f"{excess:>11.2f}%"
        f.write(row + "\n")

        # LLM-evolved heuristic row
        llm_heuristic = LLMHeuristic(best_code)
        row = f"{'LLM-Evolved':<20}"
        for dataset_name in ['OR3', 'Weibull 5k']:
            dataset = datasets[dataset_name]
            num_bins_list = []
            l1_bounds = []
            for name in dataset:
                instance = dataset[name]
                items = instance['items']
                capacity = instance['capacity']
                bins = np.array([capacity for _ in range(len(items))])
                packing, _ = online_binpack2(items, bins, lambda item, bins: llm_heuristic(item, bins)[0])
                num_bins_list.append(len(packing))
                l1_bounds.append(l1_bound(items, capacity))
            avg_bins = np.mean(num_bins_list)
            avg_l1 = np.mean(l1_bounds)
            excess = (avg_bins - avg_l1) / avg_l1 * 100
            row += f"{excess:>11.2f}%"
        f.write(row + "\n")

        f.write("\n=== LLM-Generated Heuristics Summary ===\n")
        f.write(f"{'Gen':<6}{'Round':<8}{'Score (%)':<12}{'Best':<6}{'Type':<10}\n")
        for entry in llm_code_history:
            mark = "YES" if entry['is_best'] else ""
            mtype = entry.get('mutation_type', '')
            f.write(f"{entry['generation']:<6}{entry['round']:<8}{entry['score']:<12.2f}{mark:<6}{mtype:<10}\n")
        f.write('=' * len(header) + "\n")


# --- Evolutionary bin packing heuristics ---
def evolve_binpack_heuristics(
    initial_code: str,
    dataset_name: str,
    generations: int,
    pop_size: int,
    tournament_size: int,
    api_config: dict = None,
    llm_code_history=None,
    openai_client=None
):
    dataset = datasets[dataset_name]
    opt = l1_bound_dataset(dataset)
    best_code = initial_code
    best_score = None

    # Track all candidates for reporting
    all_candidates = []

    # --- Initialize strategy history with the initial code's comment ---
    strategy_history = []
    if initial_code:
        comment = extract_strategy_comment(initial_code)
        if comment:
            strategy_history.append(comment)

    def eval_code(code, parent_score=None, parent_code=None):
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            heuristic = LLMHeuristic(code)
            num_bins_list = []
            l1_bounds = []
            error_occurred = False
            for name in dataset:
                instance = dataset[name]
                items = instance['items']  # Items are used in original order (no shuffling)
                capacity = instance['capacity']
                scores, error = heuristic(items[0], np.array([capacity]))  # quick test for error
                if error:
                    last_error = error
                    error_occurred = True
                    break
            if not error_occurred:
                for name in dataset:
                    instance = dataset[name]
                    items = instance['items']
                    capacity = instance['capacity']
                    bins = np.array([capacity for _ in range(len(items))])
                    packing, _ = online_binpack2(items, bins, lambda item, bins: LLMHeuristic(code)(item, bins)[0])
                    num_bins_list.append(len(packing))
                    l1_bounds.append(l1_bound(items, capacity))
                avg_bins = np.mean(num_bins_list)
                avg_l1 = np.mean(l1_bounds)
                excess = (avg_bins - avg_l1) / avg_l1 * 100
                return excess
            else:
                if VERBOSE_LLM_CODE:
                    print(f"\n--- LLM Generated Code (ERROR, revision requested) ---\n{code}\nError: {last_error}\n")
                print(f"LLM heuristic error: {last_error}\nRequesting revision from LLM...")
                code = llm_generate_binpack_heuristic(
                    code, parent_score if parent_score is not None else 0,
                    api_config=api_config,
                    error_message=last_error,
                    openai_client=openai_client
                )
        print(f"Failed after {max_retries} attempts. Last error: {last_error}")
        return 1e6

    # --- Initial population: seeds from good_seed.txt + invent ---
    initial_population = []
    seed_codes = []
    try:
        with open('good_seed.txt', 'r') as f:
            raw = f.read()
        # Split into blocks by blank lines
        blocks = [b.strip() for b in raw.split('\n\n') if b.strip()]
        for idx, block in enumerate(blocks):
            lines = block.split('\n')
            # First line: N) Score: X.XX%
            header = lines[0].strip()
            score = None
            if 'Score:' in header:
                try:
                    score = float(header.split('Score:')[1].split('%')[0].strip())
                except Exception:
                    score = None
            # Collect comment lines
            comment_lines = []
            code_lines = []
            in_code = False
            for line in lines[1:]:
                if line.strip().startswith('#') and not in_code:
                    comment_lines.append(line.strip())
                else:
                    in_code = True
                    code_lines.append(line)
            comment = ' '.join(comment_lines)
            code = '\n'.join(code_lines).strip()
            if code:
                seed_codes.append({'code': code, 'score': score, 'comment': comment})
    except Exception as e:
        seed_codes = []

    # Insert seeds into population
    for i, seed in enumerate(seed_codes):
        # If the score is not present, evaluate it
        score = seed['score']
        if score is None:
            score = eval_code(seed['code'])
        initial_population.append((score, seed['code'], 0, i))
        if llm_code_history is not None:
            llm_code_history.append({
                'code': seed['code'],
                'score': score,
                'generation': 0,
                'round': i,
                'is_best': False,
                'parent': None,
                'mutation_type': 'seed',
                'tournament': [],
                'elite_group': [],
                'comment': seed['comment']
            })
        if seed['comment']:
            strategy_history.append(seed['comment'])

    # Fill the rest of the population with LLM 'invent' codes
    for i in range(len(seed_codes), pop_size):
        code = llm_generate_binpack_heuristic(
            "",  # No parent code for invent
            None,
            api_config=api_config,
            openai_client=openai_client,
            mutation_type="invent",
            strategy_history=strategy_history
        )
        comment = extract_strategy_comment(code)
        if comment:
            strategy_history.append(comment)
        score = eval_code(code)
        initial_population.append((score, code, 0, i))
        if llm_code_history is not None:
            llm_code_history.append({
                'code': code,
                'score': score,
                'generation': 0,
                'round': i,
                'is_best': False,
                'parent': None,
                'mutation_type': 'invent',
                'tournament': [],
                'elite_group': []
            })

    # Set best from initial population
    initial_population.sort(key=lambda x: x[0])
    elite_population = initial_population[:pop_size]
    best_score, best_code, _, _ = elite_population[0]

    if llm_code_history is not None:
        best_code_str = best_code
        for entry in llm_code_history:
            entry['is_best'] = (entry['code'] == best_code_str)

    # --- Evolutionary loop ---
    for gen in range(1, generations + 1):
        elite_indices = [(entry[2], entry[3]) for entry in elite_population]
        for round_in_gen in range(pop_size):
            tournament = random.sample(elite_population, min(tournament_size, len(elite_population)))
            tournament_indices = [(entry[2], entry[3]) for entry in tournament]
            parent_score, parent_code, parent_gen, parent_round = min(tournament, key=lambda x: x[0])

            # Now use random invent/refine as before
            mutation_type = random.choice(["refine", "invent"])
            code_input = parent_code if mutation_type == "refine" else ""
            code = llm_generate_binpack_heuristic(
                code_input, parent_score,
                api_config=api_config,
                openai_client=openai_client,
                mutation_type=mutation_type,
                strategy_history=strategy_history
            )
            if VERBOSE_LLM_CODE:
                print(f"\n--- LLM Generated Code (Gen {gen}, Round {round_in_gen}, {mutation_type}) ---\n{code}\n")
            comment = extract_strategy_comment(code)
            if comment:
                strategy_history.append(comment)

            score = eval_code(code, parent_score=parent_score, parent_code=parent_code)
            print(f"Gen {gen}, Round {round_in_gen}, {mutation_type}: Score = {score:.2f}%")
            candidate = (score, code, gen, round_in_gen)
            all_candidates.append(candidate)
            if llm_code_history is not None:
                llm_code_history.append({
                    'code': code,
                    'score': score,
                    'generation': gen,
                    'round': round_in_gen,
                    'is_best': False,
                    'parent': (parent_gen, parent_round),
                    'mutation_type': mutation_type,
                    'tournament': tournament_indices,
                    'elite_group': elite_indices
                })
        combined = elite_population + all_candidates[-pop_size:]
        combined.sort(key=lambda x: x[0])
        elite_population = combined[:pop_size]
        if elite_population[0][0] < best_score:
            best_score, best_code, _, _ = elite_population[0]
        if llm_code_history is not None:
            best_code_str = best_code
            for entry in llm_code_history:
                entry['is_best'] = (entry['code'] == best_code_str)
        # Save progress after each generation
        save_llm_history(
            llm_code_history, best_code, best_score, datasets, l1_bound_dataset,
            first_fit_heuristic, best_fit_heuristic, discovered_heuristic_or,
            discovered_heuristic_weibull, funsearch_heuristic_weibull, LLMHeuristic
        )
        print(f"Gen {gen}: Best this gen = {elite_population[0][0]:.2f}% | Best ever = {best_score:.2f}%")
    if llm_code_history is not None:
        best_code_str = best_code
        for entry in llm_code_history:
            entry['is_best'] = (entry['code'] == best_code_str)
    return best_code, best_score
# --- Usage Example ---
llm_code_history = []
best_code, best_score = evolve_binpack_heuristics(
    initial_code="""
import numpy as np
leftover = bins - item
scores = np.full(len(bins), -np.inf, dtype=float)
fits = leftover >= 0
if np.any(fits):
    balancedness = -np.abs(leftover[fits] - bins[fits]/2) / bins[fits]
    leftover_score = -np.sqrt(leftover[fits] / (bins[fits]+1e-8))
    scores[fits] = leftover_score + balancedness
""",
    dataset_name='OR3',
    generations=15,
    pop_size=15,
    tournament_size=5,
    api_config=API_CONFIG,
    llm_code_history=llm_code_history,
    openai_client=openai_client
)


print(f"\nFull code and scores saved to llm_binpack_heuristics.txt")
