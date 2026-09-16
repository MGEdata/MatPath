import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator


def get_model_path(model_name, model_root_path=None):
    decoder_models = [
        'meta-llama/Meta-Llama-3.1-8B',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
    ]

    if model_name in decoder_models:
        return model_name

    model_paths = {
        'SteelBERT': 'SteelBERT'
    }

    if model_root_path is None:
        return model_name

    if model_name in model_paths:
        suffix = model_paths[model_name]
        full_path = os.path.join(model_root_path, suffix)
        
        if os.path.exists(os.path.join(full_path, "config.json")):
            return full_path
        
        formatted_name = model_name.replace('/', '--')
        for folder in os.listdir(model_root_path):
            if formatted_name.lower() in folder.lower() or folder.lower() in formatted_name.lower():
                check_path = os.path.join(model_root_path, folder)
                if os.path.isdir(check_path) and os.path.exists(os.path.join(check_path, "config.json")):
                    print(f"✅ Auto-discovered model at: {check_path}")
                    return check_path
    
    if os.path.exists(os.path.join(model_name, "config.json")):
        return model_name
        
    return model_name

def encode_strings(model_name: str, strings: list, batch_size: int = 16,
                   save_output: str = None, model_root_path: str = None): 
    """
    Encodes a list of strings using a specified Hugging Face model in batches.
    """
    periodic_table_symbols = set([
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
        "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ])

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    accelerator = Accelerator()
    device = accelerator.device

    resolved_model_path = get_model_path(model_name, model_root_path=model_root_path)
    print(f"Loading model from: {resolved_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=True)

    try:
        model = AutoModel.from_pretrained(
            resolved_model_path, 
            device_map="auto", 
            trust_remote_code=True
        )
    except ValueError as e:
        print(f"Warning: {e}. Loading to single device manually.")
        model = AutoModel.from_pretrained(resolved_model_path, trust_remote_code=True)
        model = model.to(device)

    model.eval()
    model = accelerator.prepare(model)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    encodings = tokenizer(strings, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False)
    dataloader = accelerator.prepare(dataloader)

    encodings_output = []

    for i, batch in enumerate(tqdm(dataloader, desc="Encoding")):
        input_batch, mask_batch = batch
        with torch.no_grad():
            outputs = model(input_ids=input_batch, attention_mask=mask_batch)
            hidden_states = outputs.last_hidden_state

            for j in range(input_batch.size(0)):
                global_idx = i * batch_size + j
                if global_idx >= len(strings):
                    break
                string = strings[global_idx]

                if string in periodic_table_symbols:
                    if hidden_states.shape[1] > 1:
                        token_embedding = hidden_states[j, 1, :]
                    else:
                        token_embedding = hidden_states[j, 0, :]
                elif 'bert' in resolved_model_path.lower():
                    token_embedding = hidden_states[j, 0, :]
                else:
                    token_embedding = hidden_states[j, :].mean(dim=0)

                encodings_output.append(token_embedding.cpu())

    encodings_output = torch.stack(encodings_output, dim=0)

    if save_output:
        torch.save(encodings_output, f"{save_output}.pt")
        print(f"Saved embeddings to {save_output}.pt")

    return encodings_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a transformer model for encoding.")
    parser.add_argument('--model', type=str, required=True, help='Model name.')
    parser.add_argument('--output', type=str, required=False, help='Output file.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--root_path', type=str, default=None, help='Root path for local models.')
    
    args = parser.parse_args()
    
    strings = ["H", "He", "Li", "Steel"]
    output = encode_strings(args.model, strings, batch_size=args.batch_size, model_root_path=args.root_path)
    print(f"Test Output Shape: {output.shape}")