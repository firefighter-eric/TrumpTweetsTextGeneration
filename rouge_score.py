from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def load_file(k):
    trump_tweets = []
    f = open(f'donald_tweets.txt', 'rb').read().decode(encoding='utf-8')
    f = f.split("\n")
    for line in f:
        l = line.split()
        if len(l) >= k:
            trump_tweets.append(l)
        else:
            trump_tweets.append([])
    return trump_tweets


def rouge_scores(k, index, generate_text, model):
    trump_tweets = load_file(k)
    initial_sentence = trump_tweets[index][:k]
    length = len(trump_tweets[index])
    hyp = generate_text(model, initial_sentence, length, ' ')
    ref = trump_tweets[index]
    rouge = Rouge()
    scores = rouge.get_scores(hyp, ref)
    for k, v in scores[0].items():
        print(str(k), v)


if __name__ == "__main__":
    hyp = "make america great again"
    ref = "make a great again"
    rouge = Rouge()
    scores = rouge.get_scores(hyp, ref)
    for k, v in scores[0].items():
        print(str(k), v)
