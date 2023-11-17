
import copy
import torch

def get_model_response(question, ground_truth, llm, tokenizer):
    rec = {
        'question' : question,
        'ground_truth' : ground_truth
    }
    inputs = tokenizer(question, return_tensors = "pt")
    outputs = llm.generate(**inputs, return_dict_in_generate = True, output_scores = True)
    prediction_answer = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens = True)[0]
    logits = get_logits(outputs["scores"])
    updated_rec = copy.deepcopy(rec)
    updated_rec["pred_answer"] = pred_ans
    
def get_logits(scores):
    all_scores = []
    for i in range(len(scores)):
        current_scores = torch.softmax(scores[i], dim = -1)
        max_score = max(current_scores[0]).detach().cpu().numpy().tolist()
        all_scores.append(max_score)
    return all_scores
    
def score_dataset(model, tokenizer, dataloader):
    for batch in dataloader:
        get_model_response(question, ground_truth)