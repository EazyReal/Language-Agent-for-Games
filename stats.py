from pathlib import Path
import json
from collections import defaultdict

def get_stats(all_experiment_results, log_file=Path('./log/main_exp/stats.json')):
    # Create a defaultdict to store rewards for each factory, id_iter, and baselines
    average_rewards = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result in all_experiment_results:
        agent_factory = result['agent_factory']
        id_iter = result['id_iter']
        baseline = result['baseline']
        reward = result['rewards'][result['agent_name']]
        average_rewards[agent_factory][id_iter][baseline].append(reward)

    # Calculate the average reward for each factory, id_iter, and baselines
    for agent_factory, id_iter_dict in average_rewards.items():
        for id_iter, baseline_dict in id_iter_dict.items():
            for baseline, reward in baseline_dict.items():
                average_reward = sum(reward) / len(reward)
                print(f"Factory: {agent_factory}, id_iter: {id_iter}, Baseline: {baseline}, Average Reward: {average_reward}")
    
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w+') as file:
            json.dump(average_rewards, file)
    except Exception as e:
        print(f"Error in writing to {log_file}: {e}")

    return average_rewards