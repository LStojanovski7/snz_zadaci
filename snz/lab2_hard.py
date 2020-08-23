# def train_classifier(classifier: DocumentClassifier, train_set: list, index_of_doc_in_tuple = 0, index_of_category_in_tuple = 1):
#     for doc in train_set:
#         classifier.train(doc[index_of_doc_in_tuple], doc[index_of_category_in_tuple])
 
 
# def print_prob_for_category(category, word, classifier):
#     prob = classifier.get_feature_per_category_probability(word, category)
#     weight_prob = classifier.weighted_probability(word, category, classifier.get_feature_per_category_probability)
#     print(word, category, round(prob, 5), round(weight_prob, 5))
 
 
# if __name__ == "__main__":
#     test_doc = input()
 
#     classifier = NaiveBayes(get_words)
#     train_classifier(classifier, train_data)
 
#     test_doc_words = list(sorted(get_words(test_doc)))
 
#     for word in test_doc_words:
#         print_prob_for_category("science", word, classifier)
#         print_prob_for_category("sport", word, classifier)
 
#     prediction = classifier.classify_document(test_doc)
#     prob = classifier.get_category_probability_for_document(test_doc, prediction)
#     prob = round(log2(prob), 5)
#     print(prediction, prob)