''' test_set_1.py

    Demo script to train and test a point-based classifier for dataset 1
    
'''

import os
from labeled_data import LabeledData
from classifier import Classifier

train = LabeledData( '../data/set_1_train.csv' )
test = LabeledData( '../data/set_1_test.csv' ) 

model = Classifier()

# Fit classifier parameters to training data:
model.fit(train)

# Plot target and clutter points from test set:
model.plot_all_points(test, fignum='Input_1', title='Test Data 1', block=False)

# Classify test points:
scores = model.classify(test)

# Plot classification results:
model.plot_results(test, scores, fignum='Result_1', block=True)

