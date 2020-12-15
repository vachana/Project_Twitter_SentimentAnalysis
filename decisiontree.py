import math
import pandas as pd
import self as self
from sklearn import model_selection

def unique_class_counts(rows):
    results = {}
    for row in rows:
        label = row[-1]
        if label not in results:
            results[label] = 0
        results[label] += 1
    return results

class GenerateQuestion:
    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def matchFeatures(self, sample):
        val = sample[self.column]
        if isinstance(val, float) or isinstance(val, int):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if isinstance(self.value, float) or isinstance(self.value, int):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))


# Calculate gini impurity using the formula
def gini(rows):
    counts = unique_class_counts(rows)
    gini_impurity = 1
    imp = [float(x) / len(rows) for x in counts.values()]
    for x in imp:
        gini_impurity -= x * x
    return gini_impurity


# Calculate entropy using the formula
def entropy(rows):
    counts = unique_class_counts(rows)
    cal_entropy = 0
    ent = [float(x) / len(rows) for x in counts.values()]
    for x in ent:
        cal_entropy += -x * math.log2(x)
    return cal_entropy

# Calculate information gain
def info_gain(left_child, right_child, currScore):
    p = float(len(left_child)) / (len(left_child) + len(right_child))
    return currScore - p * entropy(left_child) - (1 - p) * entropy(right_child)


def best_split(rows, header):
    best_gain = 0
    best_feature = None
    currScore = entropy(rows)
    noOffeatures = len(rows[0]) - 1
    true_rows, false_rows = [], []
    for col in range(noOffeatures):
        values = set([row[col] for row in rows])
        for value in values:
            question = GenerateQuestion(col, value, header)

            # Partition
            for row in rows:
                if question.matchFeatures(row):
                    true_rows.append(row)
                else:
                    false_rows.append(row)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, currScore)

            if gain >= best_gain:
                best_gain, best_feature = gain, question

    return best_gain, best_feature


def max_label(dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label

class Decision_Node:

    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows

class Leaf:

    def __init__(self, rows, id, depth):

        self.predictions = unique_class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth


def buildTree(rows, header, depth=0, id=0):

    gain, question = best_split(rows, header)
    true_rows, false_rows = [], []

    if gain == 0:
        return Leaf(rows, id, depth)

    # Partition
    for row in rows:
        if question.matchFeatures(row):
            true_rows.append(row)
        else:
            false_rows.append(row)


    true_branch = buildTree(true_rows, header, depth + 1, 2 * id + 2)
    false_branch = buildTree(false_rows, header, depth + 1, 2 * id + 1)

    return Decision_Node(question, true_branch, false_branch, depth, id, rows)

def pruneTree(node, prunedListofNodes):

    if isinstance(node, Leaf):
        return node

    if int(node.id) in prunedListofNodes:
        return Leaf(node.rows, node.id, node.depth)

    node.true_branch = pruneTree(node.true_branch, prunedListofNodes)
    node.false_branch = pruneTree(node.false_branch, prunedListofNodes)

    return node

def classification(row, node):

    if isinstance(node, Leaf):
        return node.predicted_label

    if node.question.matchFeatures(row):
        return classification(row, node.true_branch)
    else:
        return classification(row, node.false_branch)


def printTree(node, space=""):

    if isinstance(node, Leaf):
        print(space + "Leaf id: " + str(node.id) + " Predictions: " + str(node.predictions) + " Label Class: " + str(
            node.predicted_label))
        return

    # Print the question at this node
    print(space + str(node.question) + " id: " + str(node.id) + " depth: " + str(node.depth))

    # Call this function recursively on the true branch
    print(space + 'True:')
    printTree(node.true_branch, space + "  ")

    # Call this function recursively on the false branch
    print(space + 'False:')
    printTree(node.false_branch, space + "  ")


def getLeafNodes(node, leafNodes=[]):

    if isinstance(node, Leaf):
        leafNodes.append(node)
        return

    getLeafNodes(node.true_branch, leafNodes)
    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes


def getInnerNodes(node, innerNodes=[]):

    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    getInnerNodes(node.true_branch, innerNodes)
    getInnerNodes(node.false_branch, innerNodes)

    return innerNodes

def computeAccuracy(rows, node):
    count = len(rows)
    if count == 0:
        return 0
    accuracy = 0
    for row in rows:
        if row[-1] == classification(row, node):
            accuracy += 1
    return round(accuracy / count, 2)


# Main class of the decision tree
class Main:
    def __init__(self, df, header, list_data):
        self.df = df
        self.header = header
        self.list_data = list_data

    def openFile(self):
        self.df = pd.read_csv("features_test.csv",
                 names=["nouncount", "adjectivecount", "pronouncount", "adverbcount", "pos", "neg", "y"], nrows=100)
        print(self.df)
        self.header = list(self.df.columns)

    def decisionTree(self):
        # Splitting the test dataset into training and test set in 9:1 ratio respectively
        self.list_data = self.df.values.tolist()

        trainDf, testDf = model_selection.train_test_split(self.list_data, test_size=0.1)

        # Build tree from training set
        treeFromTrainSet = buildTree(trainDf, self.header)

        # Get leaf nodes of the tree from training set once the tree is built
        leafNodes = getLeafNodes(treeFromTrainSet)
        print("\nLeaf Nodes")
        for leafNode in leafNodes:
            print("Depth = " + str(leafNode.depth) + " \t \t \tID =" + str(leafNode.id))

        # Get non-leaf nodes of the tree from training set once the tree is built
        innerNodes = getInnerNodes(treeFromTrainSet)
        print("\nNon-leaf Nodes")
        for innerNode in innerNodes:
             print("Depth = " + str(innerNode.depth) + " \t \t \t ID =" + str(innerNode.id))

        # Compute maximum accuracy of the tree generated using test data
        maxAcc = computeAccuracy(testDf, treeFromTrainSet)
        print("\nAccuracy of the Tree before pruning: " + str(maxAcc * 100))

        # Print tree before performing pruning
        printTree(treeFromTrainSet)

        # Perform pruning to reduce the complexity of the final tree and increase the accuracy
        pruningId = -1
        for node in innerNodes:
            if node.id != 0:
                pruneTree(treeFromTrainSet, [node.id])
                currAcc = computeAccuracy(testDf, treeFromTrainSet)
                print("Pruned Node ID: " + str(node.id) + " to achieve accuracy: " + str(currAcc * 100) + "%")
                if currAcc > maxAcc:
                    maxAcc = currAcc
                    pruningId = node.id
                    treeFromTrainSet = buildTree(trainDf, self.header)
                if maxAcc == 1:
                    break

        if pruningId != -1:
            treeFromTrainSet = buildTree(trainDf, self.header)
            pruneTree(treeFromTrainSet, [pruningId])
            print("\nFinal Node Id to prune: " + str(pruningId))
        else:
            treeFromTrainSet = buildTree(trainDf, self.header)
            print("\nPruning did not help increase accuracy")

        # Print final tree after performing pruning
        print("\nFinal Tree with accuracy: " + str(maxAcc * 100) + "%")
        printTree(treeFromTrainSet)

Main.openFile(self)
Main.decisionTree(self)
