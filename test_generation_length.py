from glob import glob
from pandas import read_csv
from nltk.tokenize import word_tokenize
from tqdm import tqdm

expt_1_files = glob('saved_model_*/better_beam_generations_5.csv')
expt_2_files = glob('saved_model_*/better_beam_generations_human_eval.csv')

for file in expt_2_files:
    data = read_csv(file)
    avg_words = 0
    p_bar = tqdm(total=data.shape[0])
    for idx, gen in enumerate(data['pred_summary'].tolist()):
        num_words = len(word_tokenize(gen))
        avg_words = (avg_words * idx + num_words) / (idx + 1)
        p_bar.update(1)
    p_bar.close()

    print(f'Model: {file}\tGen Length:{avg_words}\n')