import pandas as pd
import os
import numpy as np


INPUT_FILE = '/data2/wanghao/Bert_ft/template.xlsx'  
OUTPUT_DIR = './datasets'                


TRAIN_RATIO = 0.80   
TEST_RATIO = 0.15    
EXP_RATIO = 0.05     


RANDOM_SEED = 42


def split_and_save():

    if not os.path.exists(INPUT_FILE):
        print(f"❌")
        return


    print(f" {INPUT_FILE} ...")
    try:

        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        return

    total_rows = len(df)

    required_cols = ['Text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️")

    common_elements = ['C', 'Mn', 'Si', 'Fe', 'Cr', 'Ni']
    found_elements = [col for col in common_elements if col in df.columns]
    if not found_elements:
        print(f"⚠️")


    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


    train_end = int(total_rows * TRAIN_RATIO)
    test_end = int(total_rows * (TRAIN_RATIO + TEST_RATIO))



    df_train = df_shuffled.iloc[:train_end].reset_index(drop=True)
    df_test = df_shuffled.iloc[train_end:test_end].reset_index(drop=True)
    df_exp = df_shuffled.iloc[test_end:].reset_index(drop=True)


    outputs = [
        ('train_data.xlsx', df_train),
        ('text_test.xlsx', df_test),
        ('exp_data.xlsx', df_exp)
    ]

    for filename, data in outputs:
        save_path = os.path.join(OUTPUT_DIR, filename)
        data.to_excel(save_path, index=False)


if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE) and os.path.exists('./template.xlsx'):
        print("⚠️")
        INPUT_FILE = './template.xlsx'
        
    split_and_save()