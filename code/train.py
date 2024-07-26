from model import Vocab, preprocess, get_batch, EncoderDecoder
from estimator import Information

from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from functools import partial
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.optim import Adam

def evaluate_and_estimate(model, analysis_dataloader, step=None, eval_dataloader=None, estimate_h=True, auto_regressive=False):
    results = {}
    
    if auto_regressive and (eval_dataloader is not None):
        eval_exact, eval_partial, eval_loss = model.auto_regressive_eval(eval_loader)
        results.update(
            {
                'loss':eval_loss,
                'exact_acc':eval_exact,
                'partial_acc':eval_partial,
                'step':step,
            }
        )
    if estimate_h:
        encodings, idx_examples = model.get_encodings(analysis_dataloader)
        results.update(
            model.h_estimate(encodings.to('cpu'), idx_examples)
        ) 
        
    model.write_log(results, 'eval')
    
    return eval_exact, eval_partial

def train(steps, data_loader, vocab, eval_loader=None, model=None, optimiser=None, eval_steps=[100]):
    if model is None:
        model =EncoderDecoder(vocab, device='cuda:0')
    if optimiser is None:
        optimiser = Adam(model.parameters())
        

    metrics = defaultdict(lambda : [])
    step = 0 
    eval_exact, eval_partial, loss = 0, 0, torch.Tensor([0.0])
    while True:
        with tqdm(total=steps) as tracker:
            tracker.set_description(f"Training | loss: {loss.item()} partial: {eval_exact} exact: {eval_partial}" )
            for i, b in enumerate(data_loader):
                
                if step in eval_steps:
                    tracker.set_description(f"Evaluation | loss: {loss.item()} partial: {eval_exact} exact: {eval_partial}" )
                    eval_exact, eval_partial = evaluate_and_estimate(
                        model, train_loader, eval_dataloader=eval_loader, step=step,
                        estimate_h=True, auto_regressive=True
                    )
                
                if step>steps: 
                    model.save(step-1)
                    return model, optimiser, metrics
                
                returned = model(b)
                model.write_log({'step':step})
                loss, exact_acc, partial_acc = returned['loss_train'], returned['exact_acc_train'], returned['partial_acc_train']
                #print(f"step {step} | loss: {loss} exact: {exact_acc} partial: {partial_acc}")
                
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                step += 1
                tracker.set_description(f"Training | loss: {loss.item()} partial: {eval_exact} exact: {eval_partial}" )
                tracker.update(1)
            
            
            
            
data = load_dataset("google-research-datasets/cfq", "mcd1")
tokens, token2index, index2token = preprocess(data['train'])
vocab = Vocab(tokens, token2index, index2token)

bs = 256
train_loader = DataLoader(
    data['train'], 
    batch_size=bs, 
    shuffle=True, 
    collate_fn=partial(get_batch, token2index=token2index)
)
eval_loader = DataLoader(
    data['test'], 
    batch_size=bs, 
    shuffle=True, 
    collate_fn=partial(get_batch, token2index=token2index)
)
        
h_estimator = Information(
    n_bins=20, 
    n_heads=64, 
    dist_fn='unit_cube', 
    smoothing_fn='discrete'
)
model =EncoderDecoder(vocab, device='cpu', h_estimator=h_estimator, d_model=512)

eval_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
eval_steps.extend(list(range(600,2600, 100)))
eval_steps.extend(list(range(2500,42000, 1000)))


model, optimiser, metrics = train(
    40001, 
    train_loader, 
    vocab, 
    eval_loader=eval_loader,
    model = model,
    eval_steps=eval_steps,
)