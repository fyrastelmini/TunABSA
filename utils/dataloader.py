
def make_train_test_data(dataset,model_name):
    from sklearn.model_selection import train_test_split
    if model_name=='BiGRU_pretrain':
        import numpy as np
        max_len = max([len(lst) for lst in dataset['tokens']])

        X_train_padded = [lst + [0] * (max_len - len(lst)) for lst in dataset['tokens']]
        X_train_padded = np.array(X_train_padded)


        max_len = max([len(lst) for lst in dataset['subject_mask_tokens']])
        subject_mask_tokens_padded = [lst + [0] * (max_len - len(lst)) for lst in dataset['subject_mask_tokens']]
        subject_mask_tokens = np.array(subject_mask_tokens_padded)

        max_len = max([len(lst) for lst in dataset['polarized_mask_tokens']])
        polarized_mask_tokens_padded = [lst + [0] * (max_len - len(lst)) for lst in dataset['polarized_mask_tokens']]
        polarized_mask_tokens = np.array(polarized_mask_tokens_padded)

        X_train, X_test, y_train_subject, y_test_subject, y_train_polarized, y_test_polarized = train_test_split(X_train_padded, subject_mask_tokens, polarized_mask_tokens, test_size=0.2, random_state=42, shuffle= True)
        return(X_train, X_test, y_train_subject, y_test_subject, y_train_polarized, y_test_polarized)
    
    elif model_name=='BiGRU_attention':
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        max_sequence_length = dataset['tokens'].apply(len).max()
        X = dataset['tokens']
        X = pad_sequences(X, maxlen=max_sequence_length, padding='post', truncating='post')

        y =dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True,stratify=y)
        return(X_train, X_test, y_train, y_test)

def preprocess(dataset,model_name,tokenizer,verbose,tokens_to_remove):
    if verbose==True:
        from tqdm import tqdm
        pbar=tqdm(total=dataset.shape[0]+2)
    if model_name=='BiGRU_pretrain':
        import ast
        dataset['subject_mask'] = dataset['subject_mask'].apply(ast.literal_eval)
        dataset['polarized_mask'] = dataset['polarized_mask'].apply(ast.literal_eval)
    
        def get_features(sentence, entity_list, polarity_list,tokenizer=tokenizer,tokens_to_remove=tokens_to_remove,verbose=verbose):
            new_indexer1=[] # to store polarity tokens
            new_indexer2=[] # to store entity tokens
            words=sentence.split(" ")
            for w, j, g in zip(words, polarity_list, entity_list):
                l = len(tokenizer(w)["input_ids"]) - 2 # remove 2&3 tokens
                for k in range(l):
                    new_indexer1.append(j)
                    new_indexer2.append(g)
            tokens = tokenizer(sentence)["input_ids"]
            for token in tokens_to_remove:
                tokens.remove(token)
            if verbose==True: pbar.update(1)

            return tokens, new_indexer2, new_indexer1
    
        list_tokens=[]
        list_subject_mask=[]
        list_polarized_mask=[]
    
        for l in range(len(dataset)):
            i, j, k = get_features(dataset.iloc[l]["sentence"], dataset.iloc[l]["subject_mask"], dataset.iloc[l]["polarized_mask"])
            list_tokens.append(i)
            list_subject_mask.append(j)
            list_polarized_mask.append(k)
        dataset["tokens"] = list_tokens
        dataset["subject_mask_tokens"] = list_subject_mask
        dataset["polarized_mask_tokens"] = list_polarized_mask
        dataset = dataset.drop(['subject_mask', 'polarized_mask'], axis=1)
        return(dataset)
    elif model_name=='BiGRU_attention':
        def get_features(sentence,tokenizer=tokenizer,tokens_to_remove=tokens_to_remove, verbose=True): 
            tokens = tokenizer(sentence)["input_ids"]
            for token in tokens_to_remove:
                tokens.remove(token)
            if verbose==True: pbar.update(1)
            return(tokens)
        dataset['tokens'] = dataset['sentence'].apply(get_features)
        dataset=dataset[["tokens","label"]]
        return(dataset)

def load_dataset(dataset_path,model_name,tokenizer,verbose=True,tokens_to_remove=[]):
    import pandas as pd
    dataset = pd.read_csv(dataset_path)
    if verbose==True: print("Preprocessing dataset:")
    dataset=preprocess(dataset, model_name, tokenizer, verbose,tokens_to_remove)
    return(dataset)