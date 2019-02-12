
# Open file containing breast cancer dataset (courtesy of UCI ML Repository)
fileref = open('C:/Users/ryjo4/Desktop/Senior Spring/AI/ProjAI_Assignment_2/breast-cancer-wisconsin.data', 'r')

# Create empty vector for storing features
breast_cancer_features = []

# Iterate through each line of file
for line in fileref:
    line = line.rstrip() # remove EOL chars
    line_vals = line.split(",") # seperate the features

    # Convert all features in the current line to integers
    # if an NA value is found (represented as '?' in dataset), exclude from
    # reature vector
    for i in range(0, len(line_vals)):
        if line_vals[i] == '?':
            break
        line_vals[i] = int(line_vals[i])
        breast_cancer_features.append(line_vals)

# Label extraction
breast_cancer_labels = []
for i in range(0, len(breast_cancer_features)):
    breast_cancer_labels.append(breast_cancer_features[i][10])

# Remove labels from feature vectors
tmp = []

# iterate through all elements in feature array
# remove last value (label)
for element in breast_cancer_features:
    newList.append(element[0:10])
breast_cancer_features = newList
