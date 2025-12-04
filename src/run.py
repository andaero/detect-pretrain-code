import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import openai
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from options import Options
from eval import *


def _compute_token_logprobs_from_ids(model, tokenizer, ids_tensor):
    """Return per-token log probabilities for the provided token ids."""
    model.eval()
    ids_tensor = ids_tensor.to(model.device)

    with torch.no_grad():
        outputs = model(ids_tensor)
        logits = outputs.logits  # (1, seq_len, vocab)
        probs = torch.nn.functional.softmax(logits[0, :-1], dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)

    input_ids = ids_tensor[0][1:].unsqueeze(-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

    mu = (probs * log_probs).sum(dim=-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    
    token_log_probs_mink_plus = (token_log_probs - mu) / sigma.sqrt()

    return token_log_probs.cpu(), token_log_probs_mink_plus.cpu()


def calculatePerplexity_local(sentence, model, tokenizer):
    """Compute perplexity-style metrics and token logprobs using the local transformer model."""
    model_inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = model_inputs.input_ids.to(model.device)
    token_log_probs, token_log_probs_mink_plus = _compute_token_logprobs_from_ids(model, tokenizer, input_ids)

    arr = token_log_probs.numpy()

    p1 = np.exp(-np.mean(arr))

    return p1, token_log_probs, token_log_probs_mink_plus, np.mean(arr)


def load_model(name1, name2):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)

    if "davinci" in name2:
        model2 = None
        tokenizer2 = None
    else:
        model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto')
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained(name2)
    return model1, model2, tokenizer1, tokenizer2

def calculatePerplexity_gpt3(prompt, modelname):
    prompt = prompt.replace('\x00','')
    responses = None
    # Put your API key here

    openai.api_key = "" # YOUR_API_KEY
    while responses is None:
        try:
            responses = openai.chat.completions.create(
                        model=modelname, 
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=0,
                        temperature=1.0,
                        logprobs=True,
                        top_logprobs=5,
                        stream=False)
        except Exception as e:
            print(f"unexpected error: {e}")
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)


     
def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2):
    pred = {}

    if "davinci" in modelname1:
        p1, all_prob, p1_likelihood = calculatePerplexity_gpt3(text, modelname1) 
        p_lower, _, p_lower_likelihood = calculatePerplexity_gpt3(text.lower(), modelname1)
    else:
        p1, all_prob, all_prob_mink_plus, p1_likelihood = calculatePerplexity_local(text, model1, tokenizer1)
        p_lower, _, all_prob_mink_plus_lower, p_lower_likelihood = calculatePerplexity_local(text.lower(), model1, tokenizer1)

    if "davinci" in modelname2:
        p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity_gpt3(text, modelname2)
    else:
        p_ref, all_prob_ref, all_prob_mink_plus_ref, p_ref_likelihood = calculatePerplexity_local(text, model2, tokenizer2)
   
   # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models (calibrate PPL to the reference model)
    pred["ppl/Ref_ppl"] = p1_likelihood-p_ref_likelihood

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    # min-k++ prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob_mink_plus)*ratio)
        topk_prob = np.sort(all_prob_mink_plus)[:k_length]
        pred[f"Min_{ratio*100}++% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred

    return ex

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, col_name, modelname1, modelname2):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2)
        all_output.append(new_ex)
    return all_output


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.target_model}_{args.ref_model}/{args.key_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and data
    model1, model2, tokenizer1, tokenizer2 = load_model(args.target_model, args.ref_model)
    print(type(model1))
    
    if "jsonl" in args.data:
        data = load_jsonl(f"{args.data}")
    else:  # load data produced by data.py (JSONL stored under the output dir)
        jsonl_path = "data/recent_wikimia/recent_wikimia.jsonl"
        dataset = load_dataset("json", data_files=str(jsonl_path))
        data = convert_huggingface_data_to_list_dic(dataset["train"])
        
        # this code reproduces the results on the original WikiMIA dataset
        # dataset = load_dataset(args.data, split=f"WikiMIA_length{args.length}")
        # data = convert_huggingface_data_to_list_dic(dataset)

    all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
    fig_fpr_tpr(all_output, args.output_dir, args.dataset_year)

