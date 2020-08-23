from document_classification import NaiveBayes, DocumentClassifier, get_words


def sample_train(cl):
    cl.train("Dali zakate da zarabotite brzi pari? Podignete ja vasata nagrada sega! Online zarabotka.", "bad")
    cl.train("""Od vasata smetka e napravena transakcija vo iznos od 1000 denari. 
            Javete se vo najbliskata ekspozitura za podetalni informacii.""", "good")
    cl.train("Online marketing! Brzi pari bez trud!", "bad")
    cl.train("""Bez da ucam dobiv desetka po ekspertni sistemi""", "good")
    cl.train("""Desetka so kromid za samo 100 denari. Vo vasata maalska kafeana!""", "bad")


def example1():
    cl = DocumentClassifier(get_words)
    sample_train(cl)

    print(cl.get_feature_counts_per_category("online", "bad"),
          cl.get_feature_per_category_probability("online", "bad"))
    print(cl.get_feature_counts_per_category("vasata", "good"),
          cl.get_feature_per_category_probability("vasata", "good"))
    print(cl.get_feature_counts_per_category("desetka", "good"),
          cl.get_feature_per_category_probability("desetka", "good"))
    print(cl.get_feature_counts_per_category("denari", "bad"),
          cl.get_feature_per_category_probability("denari", "bad"))
    print(cl.get_feature_counts_per_category("online", "good"),
          cl.get_feature_per_category_probability("online", "good"))

    print(cl.get_feature_counts_per_category("sistemi", "good"),
          cl.get_feature_per_category_probability("sistemi", "good"),
          cl.weighted_probability("sistemi", "good", cl.get_feature_per_category_probability))
    print(cl.get_feature_counts_per_category("sistemi", "bad"),
          cl.get_feature_per_category_probability("sistemi", "bad"),
          cl.weighted_probability("sistemi", "bad", cl.get_feature_per_category_probability))


def example2():
    cl = NaiveBayes(get_words)
    sample_train(cl)
    print(cl.get_category_probability_for_document('ekspertni sistemi finki', 'good'))
    print(cl.get_category_probability_for_document('ekspertni sistemi finki', 'bad'))

    cl = NaiveBayes(get_words)
    sample_train(cl)
    print(cl.classify_document('ekspertni sistemi finki', default='unknown'),
          cl.get_category_probability_for_document('ekspertni sistemi finki', 'bad'),
          cl.get_category_probability_for_document('ekspertni sistemi finki', 'good'))

    print(cl.classify_document('Sistemi i kombinacii za sigurni tipovi. Online oblozuvanje', default='unknown'),
          cl.get_category_probability_for_document('Sistemi i kombinacii za sigurni tipovi. Online oblozuvanje', 'bad'),
          cl.get_category_probability_for_document('Sistemi i kombinacii za sigurni tipovi. Online oblozuvanje',
                                                   'good'))

    print(cl.classify_document('cudna recenica za koja ne znam od koja klasa e', default='unknown'),
          cl.get_category_probability_for_document('cudna recenica za koja ne znam od koja klasa e', 'bad'),
          cl.get_category_probability_for_document('cudna recenica za koja ne znam od koja klasa e', 'good'))


if __name__ == '__main__':
    # example1()
    # example2()
    print(get_words("""Od vasata smetka e napravena transakcija vo iznos od 1000 denari. 
            Javete se vo najbliskata ekspozitura za podetalni informacii."""))


