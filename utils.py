from rouge_score import rouge_scorer
from rouge import Rouge
from nltk import PorterStemmer
import numpy as np
from simcse import SimCSE
from sacrebleu.metrics import BLEU
from passage_embedder import get_summary_similarity
from constants import DEVICE
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

SCORER_VERSION = {'ELI5': False, 'SELF': True}
SCORER_VERSION_EVAL = {'ELI5': True, 'SELF': False}
stemmer = PorterStemmer()
sim_cse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_similarity(pred_text, true_text, **kwargs):
    '''
        Used to compute similarities like ROUGE, semantic match etc
    '''
    total_score = 0
    if SCORER_VERSION['SELF']:
        # pred and true are sequeunces - string sequences
        rouge_metric_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        if not kwargs['use-fscore']: score = rouge_metric_scorer.score(true_text, pred_text)['rougeL'][1] # Returning the recall only
        else: score = rouge_metric_scorer.score(true_text, pred_text)['rougeL'][2] # Returning the F score only
        total_score += score * 100 # Returning on a 0-100 scale
    elif SCORER_VERSION['ELI5']:
        pred_text = ' '.join([stemmer.stem(word) for word in pred_text.split()])
        true_text = ' '.join([stemmer.stem(word) for word in true_text.split()])
        rouge_metric_scorer = Rouge()
        scores = rouge_metric_scorer.get_scores(pred_text, true_text)[0] # returns a list, so taking the first value only

        total_score += scores['rouge-l']['r'] * 100 # Returning the recall only, on a 0-100 scale

    if kwargs['use-sim-score']:
        total_score += sim_cse_model.similarity(pred_text, true_text) * 100
    if kwargs['use-sbert-score']:
        pred_encode = sbert_model.encode(pred_text, convert_to_tensor=True)
        true_encode = sbert_model.encode(true_text, convert_to_tensor=True)
        total_score += util.cos_sim(pred_encode, true_encode).squeeze().cpu().item() * 100 # On a -100 to 100 scale
    if kwargs['use-bleu-score']:
        max_ngram = kwargs.get('max-ngram', 3)
        bleu_metric_scorer = BLEU(lowercase=True, max_ngram_order=max_ngram, effective_order=True)
        total_score += bleu_metric_scorer.sentence_score(pred_text, [true_text]).score # Already in a 0-100 scale
    if kwargs['use-pe-score']:
        total_score += get_summary_similarity(pred_text, true_text, kwargs['pe-model-name'], kwargs['pe-model-ckpt'], DEVICE) * 100 # In 0-100 scale        
    if kwargs['len-penalize']:
        num_pred_words = len(word_tokenize(pred_text))
        num_gt_words = len(word_tokenize(true_text))
        penalization = (abs(num_pred_words - num_gt_words) / max(num_pred_words, num_gt_words))
        total_score -= penalization

    return total_score

def get_rouge_score(pred_summaries, true_summaries):
    if SCORER_VERSION_EVAL['SELF']:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1 = []
        rouge2 = []
        rougeL = []
        for pred_summary, true_summary in zip(pred_summaries, true_summaries):
            score = scorer.score(true_summary, pred_summary)
            rouge1.append(score['rouge1'][2]) # Taking the f-1
            rouge2.append(score['rouge2'][2])
            rougeL.append(score['rougeL'][2])

        return {'Rouge-1': np.average(rouge1), 'Rouge-2': np.average(rouge2), 'Rouge-L': np.average(rougeL)}
    elif SCORER_VERSION_EVAL['ELI5']:
        pred_summaries = [" ".join([stemmer.stem(word) for word in line.split()]) for line in pred_summaries]
        true_summaries = [" ".join([stemmer.stem(word) for word in line.split()]) for line in true_summaries]
        scorer = Rouge()
        scores = scorer.get_scores(pred_summaries, true_summaries, avg=True)

        return {'Rouge-1': scores['rouge-1']['r'], 'Rouge-2': scores['rouge-2']['r'], 'Rouge-L': scores['rouge-l']['r']}