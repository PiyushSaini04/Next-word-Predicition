import random
with open('data/cleaned_hinglish.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
sample = random.sample(lines, 100000)
with open('data/sample_200k.txt','w',encoding='utf-8') as out:
    out.writelines(sample)
