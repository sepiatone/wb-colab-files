import part2_intro_to_mech_interp.tests as tests
from . import submit
import torch as t


def tester():
    print("Test connection")

def test1(attn_patterns_from_shorthand: t.Tensor, attn_patterns_from_full_name: t.Tensor):
    t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
    submit.test_submit(1, mid=6, eid=1)
    print("All tests in `test1` passed!")


def test2(layer0_pattern_from_cache, layer0_pattern_from_q_and_k):
    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    submit.test_submit(2, mid=6, eid=1)
    print("All tests in `test1` passed!")

def test3(logit_attr, correct_token_logits, atol=None, rtol=None): 
    if atol is None and rtol is None:
        #this is for those who updated their test 3
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    else:
        #this is for those who did not update their test 3
        t.testing.assert_close(logit_attr, correct_token_logits, atol=atol, rtol=rtol)
    
    
    submit.test_submit(3, mid=6, eid=1)
    print("All tests in `test3` passed!")

def test4(ablation_scores, model, rep_tokens):
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
    submit.test_submit(4, mid=6, eid=1)

def test5(AB_unfactored, AB):

    t.testing.assert_close(AB_unfactored, AB)
    submit.test_submit(5, mid=6, eid=1)
    print("All tests in `test5` passed!")

def test6(full_OV_circuit, model, layer, head_index):

    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
    submit.test_submit(6, mid=6, eid=1)

def test7(decomposed_qk_input, decomposed_q, decomposed_k, rep_cache, ind_head_index):

    t.testing.assert_close(
    decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05
)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
    print("All tests in `test7` passed!")
    submit.test_submit(7, mid=6, eid=1)

def test8(decompose_attn_scores, decomposed_q, decomposed_k, model):
    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)
    submit.test_submit(8, mid=6, eid=1)

def test9(find_K_comp_full_circuit, model):
    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)
    submit.test_submit(9, mid=6, eid=1)

def test10(get_comp_score):
    tests.test_get_comp_score(get_comp_score)
    submit.test_submit(10, mid=6, eid=1)

def test11(composition_scores_batched, composition_scores):

    t.testing.assert_close(composition_scores_batched["Q"], composition_scores["Q"])
    t.testing.assert_close(composition_scores_batched["K"], composition_scores["K"])
    t.testing.assert_close(composition_scores_batched["V"], composition_scores["V"])

    submit.test_submit(11, mid=6, eid=1)
    print("All tests in `test11` passed!")