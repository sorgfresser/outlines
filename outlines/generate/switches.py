from math import comb
import torch
from torch.nn import functional as F


def switch_experts_top_k(EXPERTS, EXPERTS_PER_TOK, TOP_K, logits, allowed_tokens, probabilities_gating, experts_so_far):
    max_num = comb(EXPERTS, EXPERTS_PER_TOK)
    # Only switch experts if we haven't reached the maximum number of experts as there are no more experts to switch to
    if len(experts_so_far) >= max_num:
        return False
    probabilities = F.softmax(logits, dim=-1)
    # Check if one of k best tokens adheres to the grammar
    k_best = torch.topk(probabilities, TOP_K, dim=-1)
    for i in range(TOP_K):
        if k_best.indices[0, i] in allowed_tokens:
            print("Not switching experts as one of the top k tokens adheres to the grammar", k_best.indices[0, i])
            return False
    print("Switching experts as none of the top k tokens adheres to the grammar")
    return True


def switch_experts_top_p(EXPERTS, EXPERTS_PER_TOK, TOP_P, logits, allowed_tokens, probabilities_gating, experts_so_far):
    max_num = comb(EXPERTS, EXPERTS_PER_TOK)
    # Only switch experts if we haven't reached the maximum number of experts as there are no more experts to switch to
    if len(experts_so_far) >= max_num:
        return False
    probabilities = F.softmax(logits, dim=-1)
    # Check if one of p best tokens adheres to the grammar
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    for i in range(len(cumulative_probs)):
        if sorted_indices[i] in allowed_tokens:
            print("Not switching experts as one of the top p tokens adheres to the grammar", sorted_indices[i])
            print("At probability", sorted_probs[i], "cum:", cumulative_probs[i])
            return False
        if cumulative_probs[i] > TOP_P:
            break
    print("Switching experts as none of the top p tokens adheres to the grammar")
    return True
