import pandas as pd

# Make dataset and set multi-index
train = pd.DataFrame({'id': [1,1,2,2,3,3], 'year': [21,22,21,22,21,22], 'flowers': ['rose', 'lily', 'iris', 'tulipa', 'daffodil', 'hyacinth'], 'colors': ['red', 'white', 'blue', 'purple', 'yellow', 'pink'], 'target':[22,25,19,28,16,23]})
train.set_index(['id','year'], inplace=True)

# Unstack the dataframe: Time index is encoded into columns
unstacked = train.unstack()

# Split into training and validation samples with fraction 0.7 into training sample; random_state=0 for reproducibility

# Draw 70% into training sample
train_sample = unstacked.sample(frac=0.7, random_state=0)
# Drop training sample from whole dataframe to create validation sample
valid_sample = unstacked.drop(train_sample.index, axis=0)

# Undo the unstacking. Time is second index again.
X_train = train_sample.stack()
X_valid = valid_sample.stack()

# Now pull out the targets of train and validaton data
y_train = X_train.target
X_train.drop(['target'], axis=1, inplace=True)

y_valid = X_valid.target
X_valid.drop(['target'], axis=1, inplace=True)
