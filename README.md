Extract the twitter-datasets.zip

To build a co-occurence matrix, run the following commands. Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py

Then to calculate the word vectors:
python3 glove.py

And finally:
python3 tweet_svm.py
