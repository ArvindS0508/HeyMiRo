from pyexpat.errors import XML_ERROR_UNCLOSED_TOKEN
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
import torch
import torchtext
import nltk_sa
from nltk_sa import sa_comms
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")

uni_comm_dict = {
            "left":["left"],
            "right":["right"],
            "forward":["forward"],
            "ahead":["forward"],
            "straight":["forward"],
            "port":["left"],
            "starboard":["right"],
            "light": ["light"],
            "glow": ["light"],
            "illuminate": ["light"],
            "brighten": ["light"],
            "shine" : ["light"],
            "wagtail":["wagtail"],
            # "love":["wagtail"],
            # "good":["wagtail"],
            # "great":["wagtail"],
            # "treat": ["wagtail"],
            # "reward": ["wagtail]",
            "ear":["ear"],
            # "bad":["ear"],
            # "hate":["ear"],
            # "punishment": ["ear]"
            }

uni_comm_dict_stem =  {stemmer.stem(k):v for k, v in uni_comm_dict.items()}

bi_comm_dict = {
               ("go", "left") : ["left", "forward"],
               ("turn", "left"): ["left"],
                ("go", "right") : ["right", "forward"],
               ("turn", "right"): ["right"],
               ("go", "forward"): ["forward"]
               }
bi_comm_dict_stem =  {(stemmer.stem(k1), stemmer.stem(k2)): v for (k1, k2), v in bi_comm_dict.items()}

# print(bi_comm_dict)
# print("~"*10)
# print(bi_comm_dict_stem)

# initial sample stop words, subject to change to be more relevant to task
stop_words = ['a','in','on','at','or', 
              'to', 'the', 'of', 'an', 'by', 
              'as', 'is', 'was', 'were', 'been', 'be', 
              'for', 'this', 'that', 'these', 'those', 'if',
              'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
              'but', 'there', 'does', 'so', 've', 'their',
              'his', 'her', 'they', 'them', 'from', 'with', 'its'
              ',','.','/','?','"',"'",'!','@','#',';',':','-','*','+', '_', '=' # punctuation
             ]
# Note: punctuation here is redundant but is included just for safety
# as the pocketsphinx transcribing will not include punctuation, only words

stop_words_stem = [stemmer.stem(i) for i in stop_words]

vec_stop_words = ['a','in','on','at', 'or', 
              'to', 'the', 'of', 'an', 'by', 
              'as', 'is', 'was', 'were', 'been', 'be', 
              'are','for', 'this', 'that', 'these', 'those', 'you', 'i', 'if',
              'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
              'but', 'there', 'does', 'so', 've', 'their',
              'his', 'her', 'they', 'them', 'from', 'with', 'its'
              ',','.','/','?','"',"'",'!','@','#',';',':','-','*','+', '_', '=', # punctuation
              'please', 'make', 'turning', 'before', 'not', "don't", 'don', 'without',
              "you're", 'then', 'more', 'then', 'again', 'once', "you've", 'gone', 'made',
              'only', 'after', 'instead'
             ]

vec_stop_words_stem = [stemmer.stem(i) for i in vec_stop_words]

# negation words, subject to change
negation_words = ['not', "don't", 'don', 'without'
                 ]
negation_words_stem = [stemmer.stem(i) for i in negation_words]

temporal_words = ['before']

temporal_words_stem = [stemmer.stem(i) for i in temporal_words]

connecting_words = ['and', 'then']

connecting_words_stem = [stemmer.stem(i) for i in connecting_words]

glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus of 6 billion words
                              dim=50)   # embedding size = 100

# word = 'rat'
# other = ['dog', 'bike', 'kitten', 'puppy', 'kite', 'computer', 'neuron', 'fido', 'friend', 'canine', 'pest', 'ratatouille', 'mouse']
# for w in other:
#     dist = torch.norm(glove[word] - glove[w]) # euclidean distance
#     #dist = torch.cosine_similarity(glove[word].unsqueeze(0), glove[w].unsqueeze(0)) # euclidean distance
#     print(w, float(dist))
def glove_sim(word, word2):
    return float(torch.norm(glove[word] - glove[word2]))
    #return float(torch.cosine_similarity(glove[word].unsqueeze(0), glove[word2].unsqueeze(0)))
maxsimamount = 0.1 # 4.0
# range of ngrams
# for example, (1, 3) denotes unigrams, bigrams and trigrams
ngram_range = (1,3)

bi_flag = True

stem_lemm_flag = False # False -> Stemming, True -> Lemmatize
# lemmatization does not work yet



def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN



def parse(inp, print_flag=True, red_check=False, maxsimamount=0.1):

    # s = Sentence(inp)
    # bert_embedding = BertEmbeddings()
    # bert_embedding.embed(s)
    # for token in s:
    #     print(token)
    #     print(token.embedding)

    x_tokens = word_tokenize(inp)
    x_tokens = nltk.pos_tag(x_tokens)
    # print(x_tokens)
    x_lem = [lemmatizer.lemmatize(i, get_wordnet_pos(p)).lower() for (i, p) in x_tokens if str(i).lower() not in stop_words]
    # print(x_lem)

    x_tok= inp.split()

    x_stop = [str(i).lower() for i in x_tok if str(i).lower() not in stop_words]

    x_stem = [stemmer.stem(i) for i in x_stop]

    if stem_lemm_flag == True:
        x_proc = x_lem
    else:
        x_proc = x_stem
    
    #stemming does not handle "gone" as "go" for some reason
    for i in range(len(x_proc)):
        if x_proc[i] == 'gone':
            x_proc[i] = 'go'

    x = []

    for i in range(ngram_range[0], ngram_range[1]+1): 
        nlist = list(zip(*[x_proc[j:] for j in range(i)])) #get ngram (unigram, bigram, trigram, etc.)
        x.append(nlist)


    x_fin = x

    comms = []

    sub_comms = []

    if print_flag: print(x_fin)

    # for n_r in x_fin[0]:
    #     n = n_r[0]

    #     if n in stop_words: continue

    #     if n in comm_dict:
    #         comms.append(comm_dict[n])

    # for n_r in x_fin[2]:
    #     n = n_r

    #     neg_flag = 0

    #     for n_i in n:

    #         if n_i in stop_words or n_i in negation_words:
    #             neg_flag = 1
    #             print("DEBUG: ", n)

    #         if n_i in comm_dict:
    #             if neg_flag == 0:
    #                 comms.append(comm_dict[n_i])
    last_comm = ''
    last_temp = 0

    for n_ind in range(len(x_fin[0])):
        n_r = x_fin[0][n_ind]
        if x_fin[1] == [] or n_ind == 0:
            n_r_bi = ('',n_r[0])
        else:
            n_r_bi = x_fin[1][max(0, n_ind-1)]
        if x_fin[1] == []:
            n_r2 = x_fin[0][max(0, n_ind-2)]
        elif x_fin[2] == []:
            n_r2 = x_fin[1][max(0, n_ind-2)]
        else:
            n_r2 = x_fin[2][max(0, n_ind-2)]


        if bi_flag == False:
            n = n_r[0]
            comm_dict = uni_comm_dict
            comm_dict_stem = uni_comm_dict_stem
        else:
            n = n_r_bi
            comm_dict = bi_comm_dict
            comm_dict_stem = bi_comm_dict_stem
        n2 = n_r2

        if print_flag:
            print("~"*10, "\n")
            print(n)
            print(n2)
            print("~"*10, "\n")

        neg_flag = 0
        temp_flag = 0
        connecting_flag = 0
        found_flag = 0

        for n_i in n2:

            if n_i in negation_words_stem:
                neg_flag = 1
                if print_flag: print("DEBUG: ", n2)
            
            if n_i in temporal_words_stem:
                temp_flag = 1
            
            if n_i in connecting_words_stem and n_i != n2[min(len(n2),2)]:
                connecting_flag = 1
                if print_flag: print("DEBUG: connecting_flag", connecting_flag)

            # if n_i in comm_dict:
            #     if neg_flag == 0:
            #         comms.append(comm_dict[n_i])

        if n in stop_words_stem: continue
        if neg_flag == 0:
            if n in comm_dict_stem:
                    if temp_flag == 0:
                        if print_flag:
                            print("adding: ", n)
                        comms = comms + comm_dict_stem[n]
                        last_comm = n[0]
                        last_temp = temp_flag
                        found_flag = 1
                    else:
                        if print_flag:
                            print("adding, post: ", n)
                        sub_comms = sub_comms + comm_dict_stem[n]
                        last_comm = n[0]
                        last_temp = temp_flag
                        found_flag = 1
            else:
                # if n[0] in uni_comm_dict:
                #     if temp_flag == 0:
                #         if print_flag:
                #             print("adding: ", n)
                #         comms = comms + uni_comm_dict[n[0]]
                #     else:
                #         if print_flag:
                #             print("adding, post: ", n[0])
                #         sub_comms = comms + uni_comm_dict[n[0]]
                if n[1] in uni_comm_dict_stem:
                    conditional_flag = 0
                    if temp_flag == 0 and last_temp == 0:
                        if connecting_flag == 1:
                            conditional_flag = 1
                            if print_flag:
                                print("adding, connecting: ", n)
                            n_c = (last_comm, n[1])
                            if print_flag: print(n_c)
                            if n_c in comm_dict_stem:
                                comms = comms + comm_dict_stem[n_c]
                                found_flag = 1
                            else: conditional_flag = 0
                    else:
                        if connecting_flag == 1:
                            conditional_flag = 1
                            if print_flag:
                                print("adding, connecting: ", n)
                            n_c = (last_comm, n[1])
                            if print_flag: print(n_c)
                            if n_c in comm_dict_stem:
                                sub_comms = sub_comms + comm_dict_stem[n_c]
                                found_flag = 1
                            else: conditional_flag = 0
                    # if connecting_flag == 1:
                    #     conditional_flag = 1
                    #     if print_flag:
                    #         print("adding, connecting: ", n)
                    #     n_c = (last_comm, n[1])
                    #     if print_flag: print(n_c)
                    #     if n_c in comm_dict_stem:
                    #         comms = comms + comm_dict_stem[n_c]
                    #     else: conditional_flag = 0
                    if conditional_flag == 0:
                        if temp_flag == 0:
                            if print_flag:
                                print("adding, uni: ", n[1])
                            comms = comms + uni_comm_dict_stem[n[1]]
                            found_flag = 1
                        else:
                            if print_flag:
                                print("adding, uni, post: ", n[1])
                            sub_comms = sub_comms + uni_comm_dict_stem[n[1]]
                            found_flag = 1
            if n not in comm_dict_stem and n not in vec_stop_words_stem and found_flag == 0:
                maxsim1 = 9999
                maxsim2 = 9999
                #maxsim = -1
                maxexisting1 = ''
                maxexisting2 = ''
                for existing in comm_dict_stem.keys():
                    sim1 = glove_sim(existing[0], n[0])
                    sim2 = glove_sim(existing[1], n[1])
                    if sim1 < maxsim1 and sim2 < maxsim2 and (sim1, sim2) in comm_dict_stem: #if sim > maxsim
                        maxsim1 = sim1
                        maxexisting1 = existing[0]
                        maxsim2 = sim2
                        maxexisting2 = existing[1]
                if print_flag:
                    print("maximum simularity: ", (maxsim1, maxsim2), " for: ", (maxexisting1, maxexisting2))
                maxsim = (maxsim1 + maxsim2)/2
                maxexisting = (maxexisting1, maxexisting2)
                if maxsim <= maxsimamount: #if maxsim >= 0.9
                    if temp_flag == 0:
                        if print_flag:
                            print("adding: ", maxexisting, " based on: ", n[1])
                        comms.append(comm_dict_stem[maxexisting])
                    else:
                        if print_flag:
                            print("adding, post: ", maxexisting, " based on: ", n[1])
                        sub_comms.append(comm_dict_stem[maxexisting])
                else: #try unigram approach
                    maxsim = 9999
                    #maxsim = -1
                    maxexisting = ''
                    for existing in uni_comm_dict_stem.keys():
                        sim = glove_sim(existing, n[1])
                        if sim < maxsim: #if sim > maxsim
                            maxsim = sim
                            maxexisting = existing
                    if print_flag:
                        print("maximum simularity: ", maxsim, " for: ", maxexisting)
                    if maxsim <= maxsimamount: #if maxsim >= 0.9
                        if temp_flag == 0:
                            if print_flag:
                                print("adding: ", maxexisting, " based on: ", n[1])
                            comms.append(uni_comm_dict_stem[maxexisting])
                        else:
                            if print_flag:
                                print("adding, post: ", maxexisting, " based on: ", n[1])
                            sub_comms.append(uni_comm_dict_stem[maxexisting])

    if sub_comms != []:
        comms = comms + sub_comms
    
    if comms == []:
        comms = sa_comms(inp=inp, print_flag=print_flag);

    if red_check==True: comms = redundancy(comms)

    return comms

def redundancy(comms):
    #if print_flag:
    #print("comms: ", comms)
    true_comms = []
    direction = 90 # unit circle, 90 - forward, 0 - right, 180 - left, 270 - back
    for i in comms:
        if i == "left":
            direction+=90
        if i == "right":
            direction-=90
        if direction >= 360:
            direction-=360
        if direction < 0:
            direction +=360
        if i != "left" and i != "right":
            if direction==90: pass
            if direction==0:
                true_comms.append("right")
            if direction==180:
                true_comms.append("left")
            if direction==270:
                true_comms.append("left")
                true_comms.append("left")
            true_comms.append(i)
            direction=90
    if direction==90: pass
    if direction==0:
        true_comms.append("right")
    if direction==180:
        true_comms.append("left")
    if direction==270:
        true_comms.append("left")
        true_comms.append("left")
    return true_comms