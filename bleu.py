from nltk.translate.bleu_score import sentence_bleu


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


def bleu_score(k, index, generate_text, model):
    trump_tweets = load_file(k)
    initial_sentence = [' '.join(trump_tweets[index][:k])]
    length = len(trump_tweets[index])
    hyp = generate_text(model, initial_sentence, length, ' ')
    s = sentence_bleu(trump_tweets[index], hyp)
    print(s)


if __name__ == "__main__":
    trump_tweets = []
    f = open(f'donald_tweets.txt', 'rb').read().decode(encoding='utf-8')
    f = f.split("\n")
    for line in f:
        l = line.split()
        if len(l) >= 5:
            trump_tweets.append(l)
        else:
            trump_tweets.append([])
    bleu_score(5,10,generate_text(),model)