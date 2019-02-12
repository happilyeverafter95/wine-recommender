# Predicting Wine Quality using Text Reviews

[Accompanying Medium Article](https://medium.com/@mandygu/predicting-wine-quality-using-text-reviews-8bddaeb5285d)

## Installing Dependencies 

Download pre-trained GloVe vectors:

`wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O data/embeddings/glove.840B.300d.zip
unzip glove.840B.300d.zip`

[Download data sets from Kaggle](https://www.kaggle.com/zynicide/wine-reviews) into the same directory.

Install dependencies: 

`pip install -r requirements.txt`

Download all NLTK packages: 

`python -m nltk.downloader all`