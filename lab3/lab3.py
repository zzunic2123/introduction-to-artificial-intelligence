import csv
from collections import Counter
from math import log2

def load_data(path):
    data = []
    with open(path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data
    

class Node:
    def __init__(self, value, children):
        self.value = value
        self.children = children

class Leaf:
    def __init__(self, value):
        self.value = value
        
class Id3:
    
    def __init__(self, hiperparam=None):
        self.root= None
        self.hiperparam = hiperparam
    
    
    def fit(self, dataset):
        x = list(dataset[0].keys())[:-1]
        self.root = self.id3(dataset,dataset, x)
        print('[BRANCHES]:')
        self.print_tree(self.root)
        

    def predict(self, test_dataset):
        predictions = []
        for x in test_dataset:
            predictions.append(self.predict_part(x, self.root))
        print('[PREDICTIONS]:', ' '.join(predictions))
        return predictions

    def predict_part(self, x, root):
        if isinstance(root, Leaf):
            return root.value
        else:
            for value, child in root.children:
                if x[root.value] == value:
                    return self.predict_part(x, child)
        return 'maybe'
    

    
    def id3(self, d, d_par, X, depth=0):
        
        
        if depth == self.hiperparam:
            return Leaf(self.most_common_label(d))
            
        if not d:
            return Leaf(self.most_common_label(d_par))
        
        v = self.most_common_label(d)
        
        if not X or self.entropy(d) == 0:
            return Leaf(v)
        
        x = self.most_discriminating_feature(d,X)
        
        subtrees = []
        
        vals = set()
        for row in d:
            vals.add(row[x])

        for value in vals:
            t = self.id3(self.remove_feature_from_dataset(d,x,value),d,self.remove_feature_from_list(X,x), depth+1)
            subtrees.append((value,t))

        return Node(x,subtrees)
    

    def print_tree(self, root, path=[], depth=1):
        if isinstance(root, Leaf):
            print(' '.join(path) + ' ' + root.value)
        else:
            for value, child in root.children:
                new_path = path + [str(depth) + ":" + root.value + "=" + value]
                self.print_tree(child, new_path, depth+1)


    
    def most_discriminating_feature(self,dataset, x):
        igs = {}
        for i in x:
            igs[i] = self.information_gain(dataset,i)
            print('IG(' + i + ') = ' + str(igs[i]))
                    
        sorted_igs = dict(sorted(igs.items()))

        return max(sorted_igs, key=sorted_igs.get)
    
    def remove_feature_from_dataset(self, dataset, feature , value):
        new_dataset = []
        
        for row in dataset:
            if row[feature] == value:
                new_dataset.append(row)
        return new_dataset
    
    def remove_feature_from_list(self, X, x):
        new_list = []
        
        for k in X:
            if k != x:
                new_list.append(k)
                
        return new_list
    

    def most_common_label(self, dataset):
        labels = self.get_labels(dataset)
        labels.sort()
        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    
    def entropy(self, dataset):
        labels = self.get_labels(dataset)
        freq = Counter(labels)
        ent = 0
        for x in freq:
            prob = freq[x] / len(labels)
            ent -= prob * log2(prob)
        return ent

    def information_gain(self, dataset, feature):
        feature_values = []
        for row in dataset:
            feature_values.append(row[feature])
            
        counter = Counter(feature_values)
        init_entropy = self.entropy(dataset)
        expected_entropy = self.calculate_expected_entropy_sum(dataset, feature, counter)
        gain = init_entropy - expected_entropy
        return gain

    def calculate_expected_entropy_sum(self, dataset, feature, counter):
        sum = 0
        for val in counter:
            prob = counter[val] / len(dataset)
            subset = self.subset(dataset, feature, val)
            entropy_of_subset = self.entropy(subset)
            expected_ent = prob * entropy_of_subset
            sum += expected_ent
        return sum

    
    def subset(self, dataset, feature, value):
        subset = []
        for row in dataset:
            if row[feature] == value:
                subset.append(row)
        return subset

    def calculate_and_print_accuracy(self, dataset, predictions):
        true_labels = self.get_labels(dataset)
        correct = 0
        for true, pred in zip(true_labels, predictions):
            if true == pred:
                correct += 1
        accuracy = correct / len(true_labels)
        print('[ACCURACY]:', f'{accuracy:.5f}')
        return accuracy

    
    def get_labels(self, dataset):
        labels = []
        for row in dataset:
            last_key = list(row.keys())[-1]
            label = row[last_key]
            labels.append(label)
        return labels
    
    def calculate_confusion_matrix(self, dataset, predictions):
        labels_dataset = self.get_labels(dataset)

        unique_labels = sorted(list(set(labels_dataset)))

        num_labels = len(unique_labels)

        matrix = []
        for _ in range(num_labels):
            row = [0] * num_labels
            matrix.append(row)
            
        for true_label, predicted_label in zip(labels_dataset, predictions):
            true_index = unique_labels.index(true_label)
            predicted_index = unique_labels.index(predicted_label)
            matrix[true_index][predicted_index] += 1

        self.print_confusion_matrix(matrix)

        return matrix

    def print_confusion_matrix(self, matrix):
        print('[CONFUSION_MATRIX]:')
        for row in matrix:
            print(' '.join(map(str, row)))



import sys
def main():
    train_data = load_data(sys.argv[1])
    test_data = load_data(sys.argv[2])
    
    if len(sys.argv) == 3:
        hiperparam = None
        id3 = Id3()
    else:
        hiperparam = int(sys.argv[3])
        id3 = Id3(hiperparam)

    id3.fit(train_data)
    predictions = id3.predict(test_data)
    id3.calculate_and_print_accuracy(test_data, predictions)
    
    #if(len(set(predictions)) > 0):
    id3.calculate_confusion_matrix(test_data, predictions)
    

if __name__ == '__main__':
    main()
    
    

    
    
    

        
        