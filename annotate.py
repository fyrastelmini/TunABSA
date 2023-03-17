import pandas as pd
import ast
import sys
import warnings

warnings.filterwarnings("ignore")

df_filtered=pd.read_csv("dataset.csv")
df_filtered['word_list'] = df_filtered['word_list'].apply(ast.literal_eval)
def clear_console():
    sys.stdout.write("\033[2J\033[1;1H")
    sys.stdout.flush()
for count,i,j,lab in zip(range(df_filtered.shape[0]),df_filtered["sentence"],df_filtered["word_list"],df_filtered["label"]):
    labels_subject=[]
    labels_polarized=[]
    breaker=False
    try:
        df_out=pd.read_csv("dataset_out.csv")
    except:
        df_out=pd.DataFrame(columns=['sentence', 'label', 'subject_mask','polarized_mask'])
    if count < df_out.shape[0]: continue
    
    for l in j:
        print("***************************************************************************** \n")
        print(i,"\n")
        print("Is '",l,"' a subject (1) or a polarized word (2)")
        label=input()
        if label=="1": 
            labels_subject.append(1)
            labels_polarized.append(0)
        elif label=="2":
            labels_subject.append(0)
            labels_polarized.append(1)
        elif label=="3":
            labels_subject=[]
            labels_polarized=[]
            breaker=True
            break
        else:
            labels_subject.append(0)
            labels_polarized.append(0)
        clear_console()
    #check if first element by trying to import, if import fails means first element
    if breaker==True: continue
    line={'sentence': i, 'label': lab, 'subject_mask': labels_subject, 'polarized_mask': labels_polarized}
    df_out=df_out.append(line,ignore_index=True)
    #print(df_out)
    df_out.to_csv("dataset_out.csv",index=False)