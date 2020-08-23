from math import log, log10
from copy import deepcopy
from decision_tree_learning import build_tree,  entropy, classify, divide_set, getMaxPrediction, print_tree_custom, removeColumn, unique_counts
from document_classification import NaiveBayes, get_words, DocumentClassifier, get_words, get_words_custom
from recommender_systems import sim_distance,sim_pearson,get_recommendations,get_recommendations_item_based,transform_prefs
from tfidf import create_dataset, rank_documents,process_document,pearson, cosine, get_vocabulary

train_data = [
    ("""I like Rhythm and Blue music.""", 'medical'),
    ("""Back in my day Emo was a comedian :/""", 'real-life'),
    ("""Why sit and listen to Locke, Jack, or Syead?""", 'real-life'),
    ("""There's nothing he needs to change.""", 'medical'),
    ("""It does not exist.""", 'medical'),
    ("""I like when the Prime Minister goes door to door to find the girl!""", 'real-life'),
    ("""Mine is book by Steve Martin called 'The Pleasure of my Company'.""", 'medical'),
    ("""What differentiates a mosquitoo from a blonde?""", 'medical'),
    ("""They're pretty good. Also, that's a good song.""", 'medical'),
    ("""And every time I hear that song I get butterflies in my stomach!""", 'real-life'),
    ("""It's the biggest load of crap I've seen for ages.""", 'real-life'),
    ("""I do not think Beyonce can sing, dance, or act. You mentioned Rihanna, who is that?""", 'medical'),
    ("""as i lay dying is far far away from christ definitaly!""", 'real-life'),
    ("""I was unaware that you were in law enforcement, as well.""", 'medical'),
    ("""I might be seeing them in a few months!""", 'real-life'),
    ("""I called to say 'I Love You""", 'medical'),
    ("""that´s why they needed to open that hatch so much!""", 'real-life'),
    ("""I would most likely not vote for him, although I believe Melania would be the most attractive First Lady in our country's history.""", 'medical'),
    ("""I do not hate him.""", 'medical'),
    ("""He's supposed to be in jail!""", 'real-life'),
    ("""i thought that she did an outstanding job in the movie""", 'real-life'),
    ("""Nicole Kidman, I love her eyes""", 'real-life'),
    ("""Youtube.com also features many of the current funny ads.""", 'medical'),
    ("""I enjoy watching my companion attempt to role-play with them.""", 'medical'),
    ("""omg i love that song im listening to it right now""", 'real-life'),
    ("""Some of my favorite television series are Monk, The Dukes of Hazzard, Miami Vice, and The Simpsons.""", 'medical'),
    ("""I have a desire to produce videos on Full Metal Alchemist.""", 'medical'),
    ("""tell him you want a 3 way with another hot girl""", 'real-life'),
    ("""I would travel to that location and physically assault you at this very moment, however, I am unable to swim.""", 'medical'),
    ("""No, no, no that was WITNESS...""", 'real-life'),
    ("""aneways shonenjump.com is cool and yeah narutos awsum""", 'real-life'),
    ("""Your mother is so unintelligent that she was hit by a cup and told the police that she was mugged.""", 'medical'),
    ("""You must be creative and find something to challange us.""", 'medical'),
    ("""i think they would have, quite a shame isn't it""", 'real-life'),
    ("""I am watching it right now.""", 'medical'),
    ("""I do not know; the person who invented the names had attention deficit disorder.""", 'medical'),
    ("""im a huge green day fan!!!!!""", 'real-life'),
    ("""I believe, rather, that they are not very smart on this topic.""", 'medical'),
    ("""Of course it is Oprah, because she has been providing better advice for a longer time.""", 'medical'),
    ("""Chicken Little my son loves that movie I have to watch at least 4 times a day!""", 'real-life'),
    ("""That is the key point, that you fell asleep.""", 'medical'),
    ("""A brunette female, a blonde, and person with red hair walked down a street.""", 'medical'),
    ("""who is your best bet for american idol season five""", 'real-life'),
    ("""That is funny.  Girls need to be a part of everything.""", 'medical'),
    ("""In point of fact, Chris's performance looked like the encoure performed at a Genesis concert.""", 'medical'),
    ("""In my time, Emo was a comedian.""", 'medical'),
    ("""my age gas prices and my blood pressure  LOL""", 'real-life'),
    ("""Moriarty and so forth, but what character did the Peruvian actor portray?""", 'medical'),
    ("""What did the beaver say to the log?""", 'medical'),
    ("""Where in the world do you come up with these questions????""", 'real-life'),
    ("""even though i also agree that the girls on Love Hina are pretty scrumptious""", 'real-life'),
    ("""I miss Aaliyah, she was a great singer.""", 'medical'),
    ("""and the blond says Great they already put me on my first murder mystery case""", 'real-life'),
]

if __name__ == '__main__':

    sample = input()

    docs = [item[0] for item in train_data]
    labels = [item[1] for item in train_data]

    dataset, df, N ,vocab = create_dataset(docs, labels)

    t1 = build_tree(dataset)
    vector = process_document(sample, df, N, vocab)

    cl = NaiveBayes(get_words)

    for item in train_data:
        cl.train(item[0], item[1])

    # cl.classify_document(sample)
    # cos = rank_documents(doc, sim_func= cosine) isto i za pears
