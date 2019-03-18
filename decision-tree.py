import math
import random


class Node:
    def __init__(self, value):
        self.value = value
        self.children = {}

    def print_tree(self):
        a = [self.value]

        for key, test in self.children.items():
            a.append(self.children[key].print_tree())

        return a

    def print_tree_test(self):
        if len(self.children) == 0:
            return str(self.value)
        else:
            temp = str(self.value)

        for key, test in self.children.items():
            temp += self.children[key].print_tree()

        return temp


def read_data_text_file(file_name):
    """
    Reads the file and returns the data
    """
    file = open(file_name, 'r')
    data = []

    for line in file:
        data.append(line.rstrip('\n').split('\t'))

    return data


def get_outputs(data):
    """
    Get list of outputs from data
    """
    outputs = []
    for example in data:
        outputs.append(example[-1])
    return outputs


def is_same_output(examples):
    """
    Checks if all data have the same classification/output
    """
    outputs = get_outputs(examples)

    last_output = outputs[0]
    for output in outputs:
        if last_output is not output:
            return False
        last_output = output
    return True


def plurality_value(examples):
    """
    Get the most common output value among a set of examples, breaking ties randomly
    """
    outputs = get_outputs(examples)

    max_count = 0
    most_common_output = None
    for x in set(outputs):
        count = examples.count(x)
        if count > max_count:
            max_count = count
            most_common_output = x
    return most_common_output


def B(q):
    if q == 0 or q == 1:
        return q
    else:
        return -(q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))


def entropy(examples, attribute):
    """
    Calculate entropy of attribute
    """
    position = 0
    if len(examples) == 0:
        return 0
    for i in examples:
        if i[attribute] == examples[0][attribute]:
            position += 1
    return B(position / len(examples))


def importance(examples, attributes):
    """
    Get attribute with lowest entropy
    """
    attribute_entropy = {}
    for attribute in attributes:
        attribute_entropy[attribute] = entropy(examples, attribute)

    min_value = 10
    chosen_attribute = None

    for e in attribute_entropy:
        if attribute_entropy[e] < min_value:
            min_value = attribute_entropy[e]
            chosen_attribute = e

    return chosen_attribute


def random_attribute(attributes):
    """
    Get a random attribute
    """
    return attributes[random.randint(0, len(attributes) - 1)]


def get_attribute_value_set(examples, chosen_attribute):
    """
    Returns a set of possible values for an attribute
    """
    possible_attribute_values = []

    for example in examples:
        if example[chosen_attribute] not in possible_attribute_values:
            possible_attribute_values.append(example[chosen_attribute])
    possible_attribute_values.sort()
    return possible_attribute_values


def decision_tree_learning(examples, attributes, parent_examples, importance_enabled):
    """
    Recursive function that creates trains a decision tree based on data examples.
    """

    if not examples:
        return Node(plurality_value(parent_examples))
    elif is_same_output(examples):
        return Node(examples[0][-1])  # Return classification
    elif not attributes:
        return Node(plurality_value(examples))
    else:
        if importance_enabled:
            chosen_attribute = importance(examples, attributes)
        else:
            chosen_attribute = random_attribute(attributes)

        tree = Node(chosen_attribute)

        attribute_value_set = get_attribute_value_set(examples, chosen_attribute)

        for possible_attribute_value in attribute_value_set:
            chosen_examples = []
            for example in examples:
                if example[chosen_attribute] == possible_attribute_value:
                    chosen_examples.append(example)

            updated_attributes = list(attributes)
            updated_attributes.remove(chosen_attribute)
            subtree = decision_tree_learning(chosen_examples, updated_attributes, examples, importance_enabled)
            tree.children[possible_attribute_value] = subtree
    return tree


def classify(tree, example):
    """
    Classify example with tree
    """
    while tree.children:
        print("Example: ", example)
        print("Tree.value:", tree.value)
        print("Tree.children", tree.children)
        tree = tree.children[example[tree.value]]
    return tree.value


def test(tree, examples):
    """
    Returns results of testing for a tree
    """
    correct_count = 0
    for example in examples:
        if example[-1] == classify(tree, example):
            correct_count += 1
    print("Tests matching: " + str(correct_count) + " of " + str(len(examples)) + ". Accuracy: " + str(
        (correct_count / len(examples)) * 100) + "%")


if __name__ == '__main__':
    # Get training data and attributes
    training_data = read_data_text_file('training.txt')
    attributes = [x for x in range(len(training_data[0]) - 1)]

    # Get test data
    test_data = read_data_text_file('test.txt')

    # Random tree
    print("Random Tree")
    random_tree = decision_tree_learning(training_data, attributes, [], False)
    print(random_tree.print_tree())
    test(random_tree, test_data)

    # Importance tree
    print("\nImportance tree")
    importance_tree = decision_tree_learning(training_data, attributes, [], True)
    print(importance_tree.print_tree())
    test(importance_tree, test_data)
