from decision_tree_learning import build_tree, print_tree, classify, entropy

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]

test_cases = [['google', 'MK', 'no', 24, 'Unknown'],
              ['google', 'MK', 'no', 15, 'Unknown'],
              ['digg', 'UK', 'yes', 21, 'Unknown'],
              ['digg', 'UK', 'no', 25, 'Unknown']]

if __name__ == '__main__':
    tree = build_tree(my_data, entropy)
    for row in test_cases:
        result = classify(row, tree)
        # Opcija 1
        import operator
        class_name = max(result.items(), key=operator.itemgetter(1))[0]
        # Opcija 2
        class_name = max(result.items(), key=lambda x: x[1])[0]
        print(row)
        print(class_name)
