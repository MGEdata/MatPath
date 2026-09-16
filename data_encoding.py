#!/usr/bin/env python
# -*- coding: UTF-8 -*-
##############################################################################


import argparse
import os

if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
    del os.environ['PYTORCH_CUDA_ALLOC_CONF']
# ==========================================
import torch
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from llm_encoding import encode_strings 
import joblib
from llm_encoding import encode_strings 


def create_path(path: str, is_file: bool = False, override: bool = False):
    if is_file:
        if os.path.exists(path):
            if override:
                os.remove(path)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            pass
    else:
        if os.path.exists(path):
            if override:
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def df_stand(df):
    scaler = MinMaxScaler(feature_range=(0.00000001, 0.99999999))
    col_names = [col for col in df.columns if 'ele' in col or 'text' in col]
    if col_names:
        df[col_names] = scaler.fit_transform(df[col_names])

    if 'Tensile_value' in df.columns:
        df['Tensile_value'] = df['Tensile_value'] / 10
    if 'Yield_value' in df.columns:
        df['Yield_value'] = df['Yield_value'] / 10
    return df

def add_ele_embed(df, ele_embed_path):

    coms = list(df.loc[:, 'H':'Bi'].columns)


    com_saved_emd = pd.read_csv(ele_embed_path)
    
    if com_saved_emd.columns[0] != 'symbol':
        raise ValueError(f"{com_saved_emd.columns[0]}")
    
    symbol_col = com_saved_emd['symbol'].values

    embed_matrix = com_saved_emd.iloc[:, 1:].values 
    
    actual_dim = embed_matrix.shape[1]
    
    if actual_dim == 0:
        raise ValueError("none")


    embed_dict = {sym: vec for sym, vec in zip(symbol_col, embed_matrix)}


    element_vectors = []
    missing_elements = []
    
    for name in coms:
        if name in embed_dict:
            element_vectors.append(embed_dict[name])
        else:

            missing_elements.append(name)
            element_vectors.append(np.zeros(actual_dim))
            
    if missing_elements:

    x = np.array(element_vectors) 
    
    w_matrix = df.loc[:, 'H':'Bi'].values.astype(float)
    
    results = np.dot(w_matrix, x)
    
    final_dim = results.shape[1]
    

    if final_dim != actual_dim:
        raise ValueError(f"❌")
        
    return results

def load_data(label='train_data', ele_embed_path=None, model_name="SteelBERT", model_root_path=None):

    file_path = f'./datasets/{label}.xlsx'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    data = pd.read_excel(file_path)

    drop_cols = [
        'DOIs', 'Files', 'problem', 'status', 'Table_topic', 'title', 'abstract', 'Other_ele',
        'Text_addition', 'Tensile_name', 'Tensile_unit', 'Yield_name', 'Yield_unit',
        'Elongation_name', 'Elongation_unit', 'Material', 'Fe'
    ]
    drop_cols = [col for col in drop_cols if col in data.columns]
    data.drop(columns=drop_cols, inplace=True)

    com_cols = data.loc[:, 'H':'Bi'].columns
    data[com_cols] = data[com_cols].fillna(0.0)

    ele_embeds = add_ele_embed(data, ele_embed_path=ele_embed_path)
    
    if 'Text' not in data.columns:
        raise ValueError(f"Column 'Text' not found in {label}.xlsx")

    print(f"Encoding text for {label} using model {model_name}...")
    
    if model_root_path is None:
        model_root_path = '/model' 

    text_embeds = encode_strings(
        model_name=model_name, 
        strings=list(data['Text']), 
        batch_size=8,
        model_root_path=model_root_path
    )
    
    t_dim = text_embeds.shape[1] if hasattr(text_embeds, 'shape') else len(text_embeds[0])
    if t_dim != 768:
        raise ValueError(f"❌")

    print(f"ele_embeds.shape:{ele_embeds.shape}, text_embeds.shape: {text_embeds.shape}")

    ele_df = pd.DataFrame(ele_embeds, columns=[f'ele{i}' for i in range(ele_embeds.shape[1])])
    if hasattr(text_embeds, 'numpy'):
        text_np = text_embeds.numpy()
    else:
        text_np = np.array(text_embeds)
        
    text_df = pd.DataFrame(text_np, columns=[f'text{i}' for i in range(text_np.shape[1])])

    output_data = pd.concat([data.reset_index(drop=True), text_df, ele_df], axis=1)
    return output_data

def get_periodic_table_symbols():
    return [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    MODEL_ROOT_PATH = '/model' 
    
    parser = argparse.ArgumentParser(description="Tune hyper-parameters.")
    parser.add_argument('--model_name', type=str, default="SteelBERT", help="Model name.")
    args = parser.parse_args()
    model_name = args.model_name

    embed_path = f"./datasets/{model_name.split('/')[-1].lower()}_encodings"
    create_path(embed_path, is_file=False, override=True)

    element_symbols = get_periodic_table_symbols()
    print("Generating element embeddings...")
    ele_embeddings = encode_strings(
        model_name=model_name, 
        strings=element_symbols, 
        batch_size=8,
        model_root_path=MODEL_ROOT_PATH
    )
    
    e_dim = ele_embeddings.shape[1] if hasattr(ele_embeddings, 'shape') else len(ele_embeddings[0])
    if e_dim != 768:
    
    ele_df = pd.concat([pd.DataFrame(element_symbols, columns=["symbol"]),
                        pd.DataFrame(ele_embeddings.numpy() if hasattr(ele_embeddings, 'numpy') else ele_embeddings)], axis=1)
    ele_df.to_csv(f"{embed_path}/ele_embed.csv", encoding='utf-8', index=False)
    print(f"✅ ele_embed.csv saved. Dim: {e_dim}")


    datasets_to_process = [
        ('train_data', 'train_data_encoding.csv'), 
        ('text_test', 'new_text_data_encoding.csv'),
        ('exp_data', 'exp_data_encoding.csv')
    ]

    scaler_path = f"{embed_path}/model_scaler.pkl"
    
    if os.path.exists(scaler_path):
        os.remove(scaler_path)

    for label, filename in datasets_to_process:
        save_path = f"{embed_path}/{filename}"
        try:
            print(f"\nProcessing {label}...")
            
            data = df_stand(load_data(
                label=label, 
                ele_embed_path=f"{embed_path}/ele_embed.csv", 
                model_name=model_name,
                model_root_path=MODEL_ROOT_PATH
            ))
            
            if label == 'train_data':
                
                scaler = MinMaxScaler(feature_range=(0.00000001, 0.99999999))
                col_names = [col for col in data.columns if 'ele' in col or 'text' in col]
                scaler.fit(data[col_names]) 
                
                joblib.dump(scaler, scaler_path)



            encoding_type = 'utf-8-sig' if label == 'exp_data' else 'utf-8' 
            data.to_csv(save_path, index=False, encoding=encoding_type)
            print(f"✅ {label} saved to {save_path}. Shape: {data.shape}")
            
        except Exception as e:
            print(f"❌ Error processing {label}: {e}")
            import traceback
            traceback.print_exc()