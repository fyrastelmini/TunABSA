import transformers as tr


def get_features(sentence,polarity_list,entity_list):
    tokenizer = tr.BertTokenizer.from_pretrained("ziedsb19/tunbert_zied")
    new_indexer1=[] #to store polarity tokens
    new_indexer2=[] #to store entity tokens
    words=sentence.split(" ")
    for w,j,g in zip(words,polarity_list,entity_list):
        l=len(tokenizer(w)["input_ids"])-2 #remove 2&3 tokens
    
        for k in range(l):
            new_indexer1.append(j)
            new_indexer2.append(g)
    tokens=tokenizer(sentence)["input_ids"]
    tokens.remove(2)
    tokens.remove(3)
    return(tokens,new_indexer1,new_indexer2)