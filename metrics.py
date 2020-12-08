from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def load_file():
    trump_tweets = []
    f = open(f'donald_tweets.txt', 'rb').read().decode(encoding='utf-8')
    f = f.split("\n")
    for line in f:
        l = line.split()

        trump_tweets.append(l)

    return trump_tweets

def scores_char(k, index, generate_text, model, trump_tweets):
    initial_sentence = ' '.join(trump_tweets[index][:k])
    length = len(trump_tweets[index])
    hyp = generate_text(model, initial_sentence, length, '')
    ref = trump_tweets[index]
    ref = " ".join(ref)
    print("Generated sentence:", hyp)
    print()
    print("Reference sentence:", ref)
    print()
    print("--------------------------------------------------------------------------")
    print()
    rouge = Rouge()
    r_scores = rouge.get_scores(hyp, ref)
    print(str(k) + " initial words from #" + str(index) + " sentences -- rouge scores:")
    for key, v in r_scores[0].items():
        print(str(key), v)
    b_scores = sentence_bleu(ref.split(), hyp)
    print()
    print("--------------------------------------------------------------------------")
    print()
    print(str(k) + " initial words from #" + str(index) + " sentences -- BLEU scores:")
    print(b_scores)
    print()
    print("##########################################################################")
    print()


def scores(k, index, generate_text, model, trump_tweets):
    initial_sentence = trump_tweets[index][:k]
    length = len(trump_tweets[index])
    hyp = generate_text(model, initial_sentence, length, ' ')
    ref = trump_tweets[index]
    ref = " ".join(ref)
    print("Generated sentence:", hyp)
    print()
    print("Reference sentence:", ref)
    print()
    print("--------------------------------------------------------------------------")
    print()
    rouge = Rouge()
    r_scores = rouge.get_scores(hyp, ref)
    print(str(k) + " initial words from #" + str(index) + " sentences -- rouge scores:")
    for key, v in r_scores[0].items():
        print(str(key), v)
    b_scores = sentence_bleu(ref.split(), hyp)
    print()
    print("--------------------------------------------------------------------------")
    print()
    print(str(k) + " initial words from #" + str(index) + " sentences -- BLEU scores:")
    print(b_scores)
    print()
    print("##########################################################################")
    print()


# def bleu_scores(k, index, generate_text, model, trump_tweets):
#     initial_sentence = trump_tweets[index][:k]
#     length = len(trump_tweets[index])
#     hyp = generate_text(model, initial_sentence, length, ' ')
#     s = sentence_bleu(trump_tweets[index], hyp)
#     print(hyp)
#     print(s)


if __name__ == "__main__":
    trump_tweets = []
    # f = open(f'donald_tweets.txt', 'rb').read().decode(encoding='utf-8')
    # f = f.split("\n")
    # for line in f:
    #     l = line.split()
    #     if len(l) >= 5:
    #         trump_tweets.append(l)
    #     else:
    #         trump_tweets.append([])
    #
    # initial_sentence = trump_tweets[10][:2]
    # from metrics import *
    #
    # trump_tweets = load_file()
    # for i in (2, 4, 6, 8, 10):
    #     scores(i, 10, generate_text, model, trump_tweets)
    # for i in (9, 10, 18, 175):
    #     scores(2, i, generate_text, model, trump_tweets)
