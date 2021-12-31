1 - index and extract features
    / index_features.py

2 - cluster features into a visual vocabulary
    / cluster_features.py

3a - quantize the feature vectors to form a BOVW histogram
    for each image in the dataset
    / extract_bovw.py
OR

3b - extract pyramid bag-of-visual-words
    / extract_pbow.py

4 - train a classifier on top of the histogram representations
    / train_model_classifier.py

Test using:
    /test_model_classifier.py
