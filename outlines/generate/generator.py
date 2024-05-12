import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    import torch

    from outlines.fsm.guide import Guide


class ContextLengthExceededError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class GenerationState:
    token_ids: "torch.Tensor"
    kv_cache: "torch.Tensor"
    logits: "torch.Tensor"
    weights: "torch.Tensor"
    fsm_states: List[int]


def sequence_generator(
        model: Callable,
        sampler: Callable,
        fsms: List["Guide"],
        token_ids: "torch.Tensor",
        sequence_weights: "torch.Tensor",
        attention_masks: "torch.Tensor",
        fsm_states: List[int],
        rng: "torch.Generator",
        switch_experts: Optional[Callable] = None
) -> Iterator[GenerationState]:
    """Generates sequences of tokens.

    Parameters
    ----------
    model
        A callable that generates a probability distribution over the
        vocabulary when passed a tensor of token ids.
    sampler
        A callable that returns the next token ids, their ancestor sequence and
        the updated sequence weights when passed a distribution over the
        vocabulary.
    token_ids
        A tensor of token ids on which the sequence distribution is conditioned, of
        shape ``(n_seqs, n_prompt_tokens)``
    sequence_weights
        A tensor that contains the initial weights of the sequences, of shape
        ``(n_seqs,)``
    attention_masks
        A tensor of tensors that represent the tokens considered at the attention
        layer, of shape ``(n_seqs, n_prompt_tokens)``.
    fsms
        List of finite-state machines that drive the text generation,
        one for each sequence in the batch.
    fsm_states
        The initial states of the finite-state machine for each sequence in the batch.
    switch_experts
        Function that indicates whether the experts used in the model should be switched
        Accepts the unbiased logits, the allowed tokens and the probability distribution of the last gating network
        as well as the expert combinations already tried.

    Yields
    ------
    A new sequence.

    """

    def get_logits_biased(input_ids, kv_cache, fsms, fsm_states, experts_used=None):
        try:
            logits, kv_cache, router_logits, experts = model(input_ids, attention_masks, kv_cache, experts_used)
        except IndexError:  # Exceeding the context length
            raise ContextLengthExceededError(
                "The input length exceeds the context length of the model."
            )

        allowed_tokens = get_allowed_tokens(fsms, fsm_states)
        return logits, bias_logits(logits, allowed_tokens), kv_cache, router_logits, experts, allowed_tokens

    import torch

    if rng is None:
        rng = torch.Generator()

    kv_cache = None

    while True:
        logits, biased_logits, kv_cache, router_logits, experts, allowed_tokens = get_logits_biased(token_ids, None, fsms,
                                                                                    # todo: fix kv_cache
                                                                                    fsm_states)
        if switch_experts is not None:
            last_layer_experts = experts[
                -1]  # tensor of shape [sequence length (or 1 if past key values), experts_per_token]
            experts_tried = [([None] * last_layer_experts.shape[0]).copy() for _ in
                             range(len(experts))]  # list of shape hidden layers, sequence length
            experts_tried[-1].pop()
            experts_tried[-1].append([tuple(last_layer_experts[-1, :].tolist())])
            prob_distr = torch.nn.functional.softmax(router_logits[-1], dim=-1)[-1, :]
            while switch_experts(logits, allowed_tokens, prob_distr, experts_tried[-1][-1]):
                # Clip kv cache and token ids for speed - not working! TODO: fix
                # kv_cache = tuple([(key[..., 0:-1, :], val[..., 0:-1, :]) for key, val in kv_cache])
                # token_ids = token_ids[..., 0:-1]
                logits, biased_logits, kv_cache, router_logits, experts, allowed_tokens = get_logits_biased(token_ids,
                                                                                            None, fsms, fsm_states,
                                                                                            # todo: fix kv_cache
                                                                                            experts_tried)
                experts_tried[-1][-1].append(tuple(experts[-1][-1, :].tolist()))

        next_token_ids, ancestors, sequence_weights = sampler(
            biased_logits, sequence_weights, rng
        )

        token_ids = update_token_ids(token_ids, next_token_ids, ancestors)
        attention_masks = update_attention_masks(attention_masks, ancestors)
        kv_cache = reorder_kv_cache(kv_cache, ancestors)
        if len(ancestors) > 1:
            fsms = reorder_fsms(fsms, ancestors)
            fsm_states = reorder_fsm_states(fsm_states, ancestors)

        fsm_states = get_next_fsm_states(fsms, fsm_states, next_token_ids)
        is_finished = is_generation_finished(fsms, fsm_states)

        if is_finished:
            yield GenerationState(
                token_ids,
                kv_cache,
                logits,
                sequence_weights,
                fsm_states,
            )
            return

        yield GenerationState(
            token_ids,
            kv_cache,
            logits,
            sequence_weights,
            fsm_states,
        )


def get_next_fsm_states(
        fsms: List["Guide"], fsm_states: List[int], next_token_ids: "torch.Tensor"
) -> List[int]:
    """

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    next_token_ids
        The tokens that were just generated.

    Returns
    -------
    A `torch.Tensor` object that represents the next logit mask.

    """
    return [
        fsm.get_next_state(fsm_state, int(token_id[0]))
        for fsm, fsm_state, token_id in zip(fsms, fsm_states, next_token_ids)
    ]


def get_allowed_tokens(
        fsms: List["Guide"], fsm_states: List[int]
) -> List[Optional[Iterable[int]]]:
    """Get the new instructions for each sequence from the finite-state machine.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    A nested list that contains the ids of the logits to keep.

    """
    return [
        fsm.get_next_instruction(state).tokens for fsm, state in zip(fsms, fsm_states)
    ]


def is_generation_finished(fsms: List["Guide"], fsm_states: List[int]) -> bool:
    """Determine if the generation is finished.

    A generation is considered finished if the FSM of every sequence in the
    batch is in a final state.

    A better solution is to return finished sequences as soon as their FSM
    is in a final state.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    Whether all sequences are finished sampling.

    """
    return all([fsm.is_final_state(state) for fsm, state in zip(fsms, fsm_states)])


def update_token_ids(
        token_ids: "torch.Tensor", next_token_ids: "torch.Tensor", ancestors: "torch.Tensor"
) -> "torch.Tensor":
    """Append the sampled tokens to the running sequence of tokens.

    Parameters
    ----------
    token_ids
        The current token sequences
    next_token_ids
        The tokens that were just generated and that we need to append
        to the existing sequences.
    ancestors
        The sequences to which the token ids need to be added.

    Returns
    -------
    A new sequence of token ids that contains the tokens that were
    just generated.

    """
    import torch

    token_ids = torch.index_select(token_ids, 0, ancestors)
    return torch.concatenate([token_ids, next_token_ids], dim=-1)


def update_attention_masks(
        attention_masks: "torch.Tensor", ancestors: "torch.Tensor"
) -> "torch.Tensor":
    """Expand the attention masks.

    Parameters
    ----------
    attention_masks
        The attention masks for each sequence in the batch.
    ancestors
        The sequences to which the token ids need to be added.

    Returns
    -------
    The attention masks padded with 1s.

    """
    import torch

    attention_masks = torch.index_select(attention_masks, 0, ancestors)
    return torch.concatenate(
        [
            attention_masks,
            torch.ones(
                attention_masks.shape[:-1] + (1,), device=attention_masks.device
            ),
        ],
        axis=-1,
    )


def reorder_fsms(fsms: List["Guide"], ancestors: "torch.Tensor") -> List["Guide"]:
    reordered_fsms = []
    for ancestor in ancestors:
        reordered_fsms.append(fsms[ancestor].copy())

    return reordered_fsms


def reorder_fsm_states(fsm_states: List[int], ancestors: "torch.Tensor") -> List[int]:
    reordered_states = []
    for ancestor in ancestors:
        reordered_states.append(fsm_states[ancestor])

    return reordered_states


def reorder_kv_cache(
        kv_cache: Optional[Tuple], ancestors: "torch.Tensor"
) -> Optional[Tuple]:
    """Re-order the KV-cache based on the ancestors.

    In transformers, the object that stores the KV-cache is a tuple who elements
    are the key cache and the value cache. Each of these caches are tuples where
    each element correpond to a layer. To each layer corresponds a tensor whose
    first dimension is the batch size.

    """
    import torch

    if kv_cache is None:
        return None

    new_kv_cache: Tuple = tuple()
    for cache_item in kv_cache:
        new_cache_item: Tuple = tuple()
        for layer in cache_item:
            layer = torch.index_select(layer, 0, ancestors.to(layer.device))
            new_cache_item += (layer,)
        new_kv_cache += (new_cache_item,)

    return new_kv_cache


def bias_logits(logits: "torch.Tensor", allowed_token_ids: List) -> "torch.Tensor":
    """Mask the logits.

    The function iterates over a nested list where each list corresponds to the
    indices that need to be masked for each row in the array.

    Parameters
    ----------
    logits
        Two dimensional tensor that contains the next-token probability
        distribution.
    allowed_token_ids
        A list that contains the tokens that can be generated by the model.

    Returns
    -------
    A view of the original logits tensor where some values are masked.

    """
    import torch

    biased_logits = torch.full_like(logits, -math.inf, device=logits.device)
    for i, ids in enumerate(allowed_token_ids):
        if ids is not None:
            biased_logits[i, ids] = logits[i, ids]
        else:
            biased_logits[i] = logits[i]
    return biased_logits
