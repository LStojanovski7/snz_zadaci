from recommender_systems import sim_distance, top_matches, get_recommendations, transform_prefs, \
    get_recommendations_item_based

critics = {
    'Lisa Rose': {'Catch Me If You Can': 3.0, 'Snakes on a Plane': 3.5, 'Superman Returns': 3.5,
                  'You, Me and Dupree': 2.5, 'The Night Listener': 3.0, 'Snitch': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'The Night Listener': 3.0,
                     'You, Me and Dupree': 3.5},
    'Michael Phillips': {'Catch Me If You Can': 2.5, 'Lady in the Water': 2.5, 'Superman Returns': 3.5,
                         'The Night Listener': 4.0, 'Snitch': 2.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0,
                     'You, Me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,
                     'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Catch Me If You Can': 4.5, 'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                      'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5, 'Snitch': 4.5},
    'Toby': {'Snakes on a Plane': 4.5, 'Snitch': 5.0},
    'Michelle Nichols': {'Just My Luck': 1.0, 'The Night Listener': 4.5, 'You, Me and Dupree': 3.5,
                         'Catch Me If You Can': 2.5, 'Snakes on a Plane': 3.0},
    'Gary Coleman': {'Lady in the Water': 1.0, 'Catch Me If You Can': 1.5, 'Superman Returns': 1.5,
                     'You, Me and Dupree': 2.0},
    'Larry': {'Lady in the Water': 3.0, 'Just My Luck': 3.5, 'Snitch': 1.5, 'The Night Listener': 3.5}
}

if __name__ == '__main__':
    # Sporedba na slichnosta na Lisa Rose so ostanatite korisnici
    print(sim_distance(critics, 'Lisa Rose', 'Jack Matthews'))
    print(sim_distance(critics, 'Lisa Rose', 'Mick LaSalle'))
    print(sim_distance(critics, 'Lisa Rose', 'Claudia Puig'))
    print(sim_distance(critics, 'Lisa Rose', 'Michael Phillips'))
    print(sim_distance(critics, 'Lisa Rose', 'Gene Seymour'))
    print(sim_distance(critics, 'Lisa Rose', 'Claudia Puig'))
    print(sim_distance(critics, 'Lisa Rose', 'Toby'))

    # Naogjanje na 5 najslichni korisnici
    slicni = top_matches(critics, 'Lisa Rose')
    for critic in slicni:
        print(critic[1], '  \t%0.4f' % critic[0])

    # Preporaka na filmovi za korisnicite
    print('Preporaki spored Pirsonova korelacija')
    for critic in critics.keys():
        preporaki = get_recommendations(critics, critic)
        print('Filmovi za ', critic, ':\t', preporaki)

    # Preporaka za filmovi na Lisa Rose spored item-based pristap
    inv = transform_prefs(critics)
    print(get_recommendations_item_based(inv, 'Lisa Rose'))
