# HeyMiRo
University of Sheffield Dissertation Project 2022 - "Hey MiRo!" Making the MiRo understand natural language


Installation steps - Use the environment.yml file with Anaconda to setup the environment. Alternatively, use the pip installations specified within to setup all the libraries needed.

Additionally, downloads for NLTK will need to be done using Python and nltk.download(), which includes twitter_samples, punkt, wordnet, averaged_perceptron_tagger, stopwords. GloVe vector files will also needed to be downloaded, specifically glove.840B.300d.zip as found here: https://nlp.stanford.edu/projects/glove/

The platform used is the MiRo robot, and the operating system used in testing is Ubuntu on the University of Sheffield Diamond Laptops.

In order to use this code, create a package in catkin workspace, and place all of the files there to be used. The main file to be run from is miro_ros_version.py
(for more information on MiRo or ROS, the wikis for COM3528 and COM2009 are very helpful: https://github.com/AlexandrLucas/COM3528/wiki https://github.com/tom-howard/COM2009/wiki)

The system supports various functionality, including:
– Accounting for negation of commands, and do not execute negated commands

– Accounting for redundancy in operations and optimise the list of operations

– Accounting for temporal words which could change the order of operations, such as "before"

– Accounting for connecting words which can modify the meaning of words depending on the context, so they adopt the contextual modifiers of other words before them

– Accounting for unknown words by using similar words that are known, using word vectors to check for similar words

– Using Sentiment Analysis to provide interactivity for the robot beyond following commands

To modify the rules or to add functionality, use the nlp_bigram_ver.py file
