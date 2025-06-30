import os
from dotenv import load_dotenv
from hashlib import md5
import requests

from typing import Callable
from jaxtyping import Int, Float
import torch
from torch import Tensor
from transformer_lens import FactoredMatrix, HookedTransformer

import functools
from tqdm import tqdm
from eindex import eindex
import einops
from transformer_lens.hook_points import HookPoint

SUPABASE_URL = "https://udnmlcykctnahtnzmmmx.supabase.co"


# to-do: generate_hash() and test_submit() to be called directly from submit.py
def generate_hash(mid, eid, iid, uid):
    return md5("-".join(map(str, [mid, eid, iid, uid])).encode()).hexdigest()


def test_submit(iid, is_checked=True, mid=4, eid=1):
    load_dotenv()
    jwt_token = os.getenv("JWT")
    uid = os.getenv("UID")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not jwt_token or not uid:
        print("❌ Please login first.")
        return
    else:
        auth_headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }

        payload = {
            "id": generate_hash(mid, eid, iid, uid),
            "module_id": mid,
            "exercise_id": eid,
            "item_id": iid,
            "user_id": uid,
            "is_checked": is_checked,
        }

        url = f"{SUPABASE_URL}/rest/v1/item_completion"
        response = requests.post(url, json=payload, headers=auth_headers)
        
        if response.status_code not in [200, 201]:
            print("Insert failed:", response.json())


def get_log_probs(logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


def head_zero_ablation_hook(z: Float[Tensor, "batch seq n_heads d_head"], hook: HookPoint, head_index_to_ablate: int) -> None:
    z[:, :, head_index_to_ablate, :] = 0.0


def get_ablation_scores(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], ablation_function: Callable = head_zero_ablation_hook) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", layer), temp_hook_fn)])
            # Calculate the loss difference (= negative correct logprobs), only on the last `seq_len` tokens
            loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1) :].mean()
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


def test_get_ablation_scores(ablation_scores: Float[Tensor, "layer head"], model: HookedTransformer, rep_tokens: Float[Tensor, "batch seq"]):
    ablation_scores_expected = get_ablation_scores(model, rep_tokens)
    torch.testing.assert_close(ablation_scores, ablation_scores_expected)


def test_full_OV_circuit(OV_circuit: FactoredMatrix, model: HookedTransformer, layer: int, head: int):
    W_E = model.W_E
    W_OV = FactoredMatrix(model.W_V[layer, head], model.W_O[layer, head])
    W_U = model.W_U

    OV_circuit_expected = (W_E @ W_OV) @ W_U
    
    assert isinstance(OV_circuit_expected, FactoredMatrix)
    torch.testing.assert_close(OV_circuit.get_corner(20), OV_circuit_expected.get_corner(20))


def decompose_attn_scores(decomposed_q: Float[Tensor, "q_comp q_pos d_head"], decomposed_k: Float[Tensor, "k_comp k_pos d_head"], model: HookedTransformer) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    """
    return einops.einsum(
        decomposed_q,
        decomposed_k,
        "q_comp q_pos d_head, k_comp k_pos d_head -> q_comp k_comp q_pos k_pos",
    ) / (model.cfg.d_head**0.5)


def test_decompose_attn_scores(decompose_attn_scores_ip: Callable, q: torch.Tensor, k: torch.Tensor, model: HookedTransformer):
    decomposed_scores = decompose_attn_scores_ip(q, k, model)
    decomposed_scores_expected = decompose_attn_scores(q, k, model)

    torch.testing.assert_close(decomposed_scores, decomposed_scores_expected)


def find_K_comp_full_circuit(model: HookedTransformer, prev_token_head_index: int, ind_head_index: int) -> FactoredMatrix:
    """
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side (direct from token
    embeddings) and the second dimension being the key side (going via the previous token head).
    """
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]

    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K
    
    return FactoredMatrix(Q, K.T)


def test_find_K_comp_full_circuit(find_K_comp_full_circuit_ip: Callable, model: HookedTransformer):
    K_comp_full_circuit: FactoredMatrix = find_K_comp_full_circuit_ip(model, 7, 4)
    K_comp_full_circuit_expected: FactoredMatrix = find_K_comp_full_circuit(model, 7, 4)

    assert isinstance(K_comp_full_circuit, FactoredMatrix), "Should return a FactoredMatrix object!"
    
    torch.testing.assert_close(
        K_comp_full_circuit.get_corner(20),
        K_comp_full_circuit_expected.get_corner(20),
        atol=1e-4,
        rtol=1e-4,
    )


def get_comp_score(W_A: Float[Tensor, "in_A out_A"], W_B: Float[Tensor, "out_A out_B"]) -> float:
    """
    Return the composition score between W_A and W_B.
    """
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item()


def test_get_comp_score(get_comp_score_ip: Callable):
    W_A = torch.rand(3, 4)
    W_B = torch.rand(4, 5)

    comp_score = get_comp_score_ip(W_A, W_B)
    comp_score_expected = get_comp_score(W_A, W_B)

    torch.testing.assert_close(comp_score, comp_score_expected)


# def tester():
#    print("Test connection")


def test1(attn_patterns_from_shorthand: torch.Tensor, attn_patterns_from_full_name: torch.Tensor):
    torch.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
    test_submit(1, mid=6, eid=1)
    print("All tests in `test1` passed!")


def test2(layer0_pattern_from_cache, layer0_pattern_from_q_and_k):
    torch.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    test_submit(2, mid=6, eid=1)
    print("All tests in `test2` passed!")


def test3(logit_attr, correct_token_logits, atol=None, rtol=None): 
    if atol is None and rtol is None:
        #this is for those who updated their test 3
        torch.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    else:
        #this is for those who did not update their test 3
        torch.testing.assert_close(logit_attr, correct_token_logits, atol=atol, rtol=rtol)    
    
    test_submit(3, mid=6, eid=1)
    print("All tests in `test3` passed!")


def test4(ablation_scores, model, rep_tokens):
    test_get_ablation_scores(ablation_scores, model, rep_tokens)
    submit.test_submit(4, mid=6, eid=1)


def test5(AB_unfactored, AB):
    torch.testing.assert_close(AB_unfactored, AB)
    test_submit(5, mid=6, eid=1)
    print("All tests in `test5` passed!")


def test6(full_OV_circuit, model, layer, head_index):
    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
    test_submit(6, mid=6, eid=1)
    print("All tests in `test6` passed!")


def test7(decomposed_qk_input, decomposed_q, decomposed_k, rep_cache, ind_head_index):
    torch.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    torch.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    torch.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)

    test_submit(7, mid=6, eid=1)
    print("All tests in `test7` passed!")
    

def test8(decompose_attn_scores, decomposed_q, decomposed_k, model):
    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)
    test_submit(8, mid=6, eid=1)
    print("All tests in `test8` passed!")


def test9(find_K_comp_full_circuit, model):
    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)
    test_submit(9, mid=6, eid=1)
    print("All tests in `test5` passed!")
    

def test10(get_comp_score):
    tests.test_get_comp_score(get_comp_score)
    test_submit(10, mid=6, eid=1)
    print("All tests in `test10` passed!")


def test11(composition_scores_batched, composition_scores):
    torch.testing.assert_close(composition_scores_batched["Q"], composition_scores["Q"])
    torch.testing.assert_close(composition_scores_batched["K"], composition_scores["K"])
    torch.testing.assert_close(composition_scores_batched["V"], composition_scores["V"])

    test_submit(11, mid=6, eid=1)
    print("All tests in `test11` passed!")
