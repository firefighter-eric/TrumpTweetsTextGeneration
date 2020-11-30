import pandas as pd


def remove_url(line):
    out = []
    for x in line.split():
        if 'http' not in x:
            out.append(x)
    return ' '.join(out)


df = pd.read_csv('Donald-Tweets!.csv')
f = open('donald_tweets.txt', 'w', encoding='utf-8')
for line in df.Tweet_Text:
    f.write(remove_url(line) + '\n')
f.close()
