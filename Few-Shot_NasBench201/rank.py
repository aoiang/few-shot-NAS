import json
from scipy import stats


with open('./supernet_info/nasbench201', 'r') as f:
    ground_truth = json.load(f)

with open('./supernet_info/few-shot-supernet', 'r') as f:
    few_shot = json.load(f)

with open('./supernet_info/one-shot-supernet', 'r') as f:
    one_shot = json.load(f)


ranked_ground_truth_list = sorted(ground_truth.items(), key=lambda d: d[1], reverse=True)
ranked_one_shot_list = sorted(one_shot.items(), key=lambda d: d[1], reverse=True)
ranked_few_shot_list = sorted(few_shot.items(), key=lambda d: d[1], reverse=True)


one_shot_real_rank = []
few_shot_real_rank = []


ranked_ground_truth_acc = []
ranked_one_shot_acc = []
ranked_few_shot_acc = []



for i in range(len(ranked_ground_truth_list)):

    ranked_ground_truth_acc.append(ranked_ground_truth_list[i][1])
    ranked_one_shot_acc.append(ranked_one_shot_list[i][1])
    ranked_few_shot_acc.append(ranked_few_shot_list[i][1])

    one_shot_real_rank.append(one_shot[ranked_ground_truth_list[i][0]])
    few_shot_real_rank.append(few_shot[ranked_ground_truth_list[i][0]])



tau_one_shot, p_value_one_shot = stats.kendalltau(ranked_one_shot_acc, one_shot_real_rank)
print("kendall_tau for one_shot is",  tau_one_shot)

tau_few_shot, p_value_few_shot = stats.kendalltau(ranked_few_shot_acc, few_shot_real_rank)
print("kendall_tau for few_shot is", tau_few_shot)

