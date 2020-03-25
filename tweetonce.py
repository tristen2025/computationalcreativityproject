import sys
import glob
import traceback
import os
import wget
import tweepy
import markovify
import random
import re
from nltk.tag import pos_tag
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from csv import reader
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import scipy
from datetime import datetime
import cv2
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import nltk

#encodings=None
#encodings = encodings if encodings is not None else ['utf-8', 'ISO-8859-1']
#f= open(sys.argv[1], 'r', encoding=encoding)
class TwitterAPI:
  def __init__(self):# initialize the keys for the twitter api
    print("running init")
    consumer_key=''# add the api keys
    consumer_secret=''
    auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
    access_token=''
    access_token_secret=''
    auth.set_access_token(access_token,access_token_secret)
    self.api= tweepy.API(auth)
    
  def tweet_function(self,message):# tweets text
    self.api.update_status(status=message)
  def tweet_media(self,message,picture):# tweets text and a picture
    self.api.update_with_media(filename=picture,status=message)
  def create_folder(self,output_folder):
      if not os.path.exists(output_folder):
          os.makedirs(output_folder)
  def tweet_media_urls(self,tweet_status):
      media = tweet_status._json.get('extended_entities', {}).get('media', [])
      if (len(media) == 0):
        return []
      else:
        return [item['media_url'] for item in media]

  def download_images(self,status, num_tweets, output_folder):
      self.create_folder(output_folder)
      downloaded = 0
      try:

          for tweet_status in status:

            if(downloaded >= num_tweets):
              break

            for count, media_url in enumerate(self.tweet_media_urls(tweet_status)):
              # Only download if there is not a picture with the same name in the folder already
              created = tweet_status.created_at.strftime('%d-%m-%y at %H.%M.%S')
              file_name = "newimage_({}).jpg".format(count+1)
              if not os.path.exists(os.path.join(output_folder, file_name)):
                  try:
                   wget.download(media_url +":orig", out=str(output_folder+'/'+file_name))
                   downloaded += 1
                  except:
                      print("error")
      except:
           print("error downloading")
      print("exiting")       
  def download_images_by_user(self, username, retweets, replies, num_tweets, output_folder):
      status = tweepy.Cursor(self.api.user_timeline, screen_name=username, include_rts=retweets, exclude_replies=replies, tweet_mode='extended').items()
      self.download_images(status, num_tweets, output_folder)


  def download_images_by_tag(self, tag, retweets, replies, num_tweets, output_folder):
      status = tweepy.Cursor(self.api.search, '#'+tag, include_rts=retweets, exclude_replies=replies, tweet_mode='extended').items()
      self.download_images(status, num_tweets, output_folder)
class myMarkov:
    tree=dict()
    propernouns=[]
    commonnouns=[]
    words=[]
    reccount=0
    def __init__(self):
        self.tree=dict()
        self.propernouns=[]
        self.commonnouns=[]
        self.words=[]
        self.reccount=0

        with open(sys.argv[1]) as f:
            lines = f.readlines()
        for el in lines:
            print(el)
            print(type(el))
            regex = re.compile('[^a-zA-Z ]')
            #el = [regex.sub('',w) for w in el]
            el1=regex.sub(' ',el)
            tagged_sent = pos_tag(el1.split())
            print(tagged_sent)
            self.propernouns=self.propernouns+[word for word,pos in tagged_sent if pos == 'NNP']

            self.commonnouns=self.commonnouns+[word for word,pos in tagged_sent if pos == 'NN']
        print("common")
        print(self.commonnouns) 
        print("proper")
        print(self.propernouns)
        regex = re.compile('[^a-zA-Z ]')
        lines = [regex.sub('',w) for w in lines]
        f= open(sys.argv[1], 'r')
        words = filter(lambda s: len(s) > 0, re.split(r'[\s"]', f.read()))
        print(words)
        words = [w.lower() for w in words]
        print(words)
        #regex = re.compile('.')
        #for ind,w in enumerate(list(words)):
        ind=0
        for w in words:
            #print("appending period "+str(ind)+" "+str(w))
            if w=='.':
                print("char is period")
                #continue
            elif '.' in w: 
                words.insert(ind+1,'.')
                if ind < len(words): 
                    words[ind]=w.replace('.','')
                    print("appending period "+str(ind)+" "+str(w)+" "+str(words[ind])+str(words[ind+1]))
                    #ind=ind+1
            #words[ind]=w.replace('.','')
            #print("appending period "+str(ind)+" "+str(w)+" "+str(words[ind]))
            ind=ind+1

        regex = re.compile('[^a-zA-Z\.]')
        words = [regex.sub('',w) for w in words]
        print(words)
        for ind,w in enumerate(words):
            if w in self.commonnouns and words.count(w)<2:
                words[ind]='sharks'
            elif w in self.propernouns and words.count(w)<2:
                words[ind]='Donald_Duck'
        for a, b in [(words[i], words[i + 1]) for i in range(len(words) - 1)]:
            if a not in self.tree:
                self.tree[a]=dict()
            if b not in self.tree[a]:
                self.tree[a][b]=1
            elif b in self.tree[a]:
                self.tree[a][b]=self.tree[a][b]+1
        self.commonnouns.append('sharks')
        self.propernouns.append('Donald_Duck')
            
        '''    
        for a, b, c in [(words[i], words[i + 1],words[i+2]) for i in range(len(words) - 2)]:
            p1=(a,b)
            print(p1)
            if p1 not in self.tree:
                self.tree[p1]=dict()
            if c not in self.tree[p1]:
                self.tree[p1][c]=1
            elif c in self.tree[p1]:
                self.tree[p1][c]=self.tree[p1][c]+1
        '''         


        for key in self.tree:
            print(str(key)+" "+str(self.tree[key]))
    def graph(self):
        fromlist=[]
        tolist=[]
        for el in self.tree:
            for el2 in  self.tree[el]:
                fromlist.append(el)
                tolist.append(el2)
        #print(fromlist)        
        #print(tolist)        
        df = pd.DataFrame({ 'from':fromlist, 'to':tolist})
        G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph() )

        # Make the graph
        pos = nx.spring_layout(G,scale=100,k=1,iterations=20)
        nx.draw(G, layout=pos,with_labels=True,node_size=100, alpha=0.3, arrows=True)
        plt.axis("off")
        #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
        #pos = nx.spring_layout(G,k=0.6,scale=4)
        #plt.show()
        #plt.draw()
        plt.savefig("graph.png",dpi=300)
        plt.show()
        #plt.show()
        plt.close()
    #from datetime import datetime
    def recurse(self,res,k1,beento):
        while k1 =='.' or len(self.tree[k1])<1:
            return res
            k1=random.choice(list(self.tree.keys()))
        random.seed(datetime.now())
        print(res)
        problist=[]
        #if len(res)>200:
            #return res
        beento.append(k1)


        for key in self.tree[k1]:
            if len(self.tree[key])<0: continue
                #res=res+key
                #i=100000
                #problist=[]
                #break
            if  key in self.tree and len(self.tree[key])>0 and key not in beento: problist.append((key,random.randint(0,(self.tree[k1][key]+len(self.tree[key])+1)*5)))

        print(problist)
        if len(problist)==0: 
            #print(res)
            return res
        maxscore=-1
        maxstr=''
        for el in problist:
            if el[1]>maxscore:
                maxstr=el[0]
        #print(res) 
        sys.stderr.write(str(problist)+"\n")
        if len(res)>120:
            for el in problist:
                if el[0]=='.':
                    #res=res+'.'
                    return res


        start=maxstr
        start=(random.choice(problist))[0]
        
        res=res+" "+start
        print("res is "+str(res))
        beento2=list(beento)
        newres=self.recurse(res,maxstr,beento2)
        print("new res is "+str(newres))
        self.reccount=self.reccount+1
        if self.reccount >150:
            return newres
        while abs(len(newres)-len(res))<15 and len(newres)<30:
            random.seed(datetime.now())
            newres=self.recurse(res,(random.choice(problist))[0],beento2)
            print("new res is "+str(newres))
        res=newres
        return newres

    
    def generate(self):
            '''
            import random   
            for a, b in [(words[i], words[i + 1]) for i in range(len(words) - 1)]:
                p1=(a,b)
                print(p1)
                if p1 not in tree: continue
                problist=[]
                for key in tree[p1]:
                    if  (p1[1],key) in tree and len(tree[(p1[1],key)])>0:problist.append((key,random.randint(0,tree[p1][key]+1)))
                print(problist)
                if len(problist)==0:
                    break
                maxscore=0
                maxstr=''
                for el in problist:
                    if el[1]>maxscore:
                        maxstr=el[0]
                print(maxstr)     
             '''   
            #print(len(self.tree))
            random.seed(datetime.now())

            start=random.choice(list(self.tree.keys()))
            #print(start)
            #while len(self.tree[start])<1 and (start in self.commonnouns or start in self.propernouns):
                #print(start)
                #start=random.choice(list(self.tree.keys()))
            while len(self.tree[start])<1 :
                #print(start)
                start=random.choice(list(self.tree.keys()))
            res=start
            self.reccount=0
            while len(res)<150:
                self.reccount=0
                start=random.choice(list(self.tree.keys()))
                res=self.recurse(res,start,[])
            regex=re.compile('sharks') 
            count=0
            while 'sharks' in res and count<5:
                count=count+1
                #print("replacing")
                #regex.sub(random.choice(self.commonnouns), res, 1)
                res=res.replace('sharks',random.choice(self.commonnouns),1)
            regex=re.compile('Donald_Duck')    
            count=0
            while 'Donald_Duck' in res and count<5:
                count=count+1
                #print("replacing")
                #regex.sub(random.choice(self.propernouns), res, 1)
                res=res.replace('Donald_Duck',random.choice(self.propernouns),1)
            return res
            #print("loop")
            i=0
            beento=[]
            while i<500:
                i=i+1
                p1=start
                beento.append(p1)
                #print(p1)
                if p1 not in self.tree: continue
                problist=[]
                for key in self.tree[p1]:
                    if len(self.tree[key])<0: continue
                    #if i > 50 and key=='.': 
                        #res=res+key
                        #i=100000
                        #problist=[]
                        #break
                    if  key in self.tree and len(self.tree[key])>0 and key not in beento: problist.append((key,random.randint(0,self.tree[p1][key]+len(self.tree[key]))+1))
                #print(problist)
                #print(len(problist))
                if len(problist)==0: 
                    #print(res)
                    break
                maxscore=-1
                maxstr=''
                for el in problist:
                    if el[1]>maxscore:
                        maxstr=el[0]
                #print(res) 
                start=maxstr
                
                res=res+" "+start
            regex=re.compile('sharks') 
            count=0
            while 'sharks' in res and count<5:
                count=count+1
                #print("replacing")
                #regex.sub(random.choice(self.commonnouns), res, 1)
                res=res.replace('sharks',random.choice(self.commonnouns),1)
            regex=re.compile('Donald_Duck')    
            count=0
            while 'Donald_Duck' in res and count<5:
                count=count+1
                #print("replacing")
                #regex.sub(random.choice(self.propernouns), res, 1)
                res=res.replace('Donald_Duck',random.choice(self.propernouns),1)
            return res    

class sentence_predictor:
    #vectorizer = CountVectorizer(min_df=0, lowercase=False)
    #vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer = CountVectorizer(ngram_range=(0,2),min_df=0, lowercase=False)
    lines=[]
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    le = LabelEncoder()
    def __init__(self):
        #lines=[]
    #vectorizer = CountVectorizer(min_df=0, lowercase=False)
        #with open(sys.argv[1]) as f:
            #lines = f.readlines()
        lines=[]
        #with open(sys.argv[1]) as f:
            #lines = f.readlines()
        with open(sys.argv[1]) as f:
            text = f.read()
            lines=re.split(r' *[\.\?!][\'"\)\]]* *', text)+f.readlines()
        security=[]
        with open(sys.argv[2]) as f:
            security = f.readlines()
        with open(sys.argv[3]) as f:
            tech = f.readlines()
        with open(sys.argv[4]) as f:
            fantasy = f.readlines()
        regex = re.compile('[^a-zA-Z ]')
        lines = [regex.sub('',w) for w in lines]
        self.vectorizer.fit(lines)
        labels=[]
        trainsec=self.vectorizer.transform(security).toarray()
        print("train sec is ")
        print(trainsec)
        for el in security:
            labels.append("sec")
        traintech=self.vectorizer.transform(tech).toarray()
        print("train sec is ")
        print(traintech)

        for el in tech:
            labels.append("tech")
        for el in fantasy:
            labels.append("fantasy")
        #self.vectorizer = CountVectorizer(min_df=0, lowercase=False)
        #vectorizer.fit(lines)
        self.json_file = open('model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")

        self.le = LabelEncoder()
        self.le.classes_ = np.load('textclasses.npy')


        for el in security:
            print(self.vectorizer.transform([el]).toarray())
            pre=self.vectorizer.transform([el]).toarray()
            prediction=self.model.predict_classes([pre])
            print(self.le.inverse_transform(prediction))
        for el in tech:
            print(self.vectorizer.transform([el]).toarray())
            pre=self.vectorizer.transform([el]).toarray()
            prediction=self.model.predict_classes([pre])
            print(self.le.inverse_transform(prediction))
        for el in fantasy:
            print(self.vectorizer.transform([el]).toarray())
            pre=self.vectorizer.transform([el]).toarray()
            prediction=self.model.predict_classes([pre])
            print(self.le.inverse_transform(prediction))
    def predict(self,str1):
        pre=self.vectorizer.transform([str1]).toarray()
        prediction=self.model.predict_classes([pre])
        ret1=self.le.inverse_transform(prediction)
        return ret1[0]
    def predict_num(self,str1):
        pre=self.vectorizer.transform([str1]).toarray()
        return self.model.predict([pre])
        #ret1=self.le.inverse_transform(prediction[0])
        #return ret1[0]

class imagePredictor:
    json_file = open('image_test1_vgg.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    encoder = LabelEncoder()
    def __init__(self):
        self.json_file = open('image_test1_vgg.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.loaded_model_json)
        # load weights into new model
        self.model.load_weights("image_testw_vgg.h5")
        print("Loaded model from disk")
    def predict(self,str1):    
        print("Loaded model from disk")
        image = cv2.imread(str1)
        imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(0,255,0),3)
        print(image)
        image = cv2.resize(image, (100, 100))
        print(image)
        image = np.array(image, dtype="float") / 255.0
        print(image)
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('imageclasses_vgg.npy')
        predictions = self.model.predict_classes([[image]])
        print(predictions)
        print(str(self.encoder.inverse_transform([predictions])))
        return self.encoder.inverse_transform([predictions])[0]
class myTextGen:
    tokenizer = Tokenizer()
    data = open(sys.argv[1]).read()
    json_file = open('text_2_model.json', 'r')
    loaded_model_json = json_file.read()
    #json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    #model.load_weights('text_2_model.h5')
    #def dataset_preparation(data):
    predictors='' 
    label='' 
    max_sequence_len='' 
    total_words='' 
    def __init__(self):
        data = open(sys.argv[1]).read()
        self.json_file = open('text_2_model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.loaded_model_json)
        # load weights into new model
        self.model.load_weights('text_2_model.h5')

        # basic cleanup
        corpus = data.lower().split("\n")

        # tokenization  
        self.tokenizer.fit_on_texts(corpus)
        total_words = len(self.tokenizer.word_index) + 1

        # create input sequences using list of tokens
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        # pad sequences 
        self.max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre'))

        # create predictors and label
        self.predictors, self.label = input_sequences[:,:-1],input_sequences[:,-1]
        self.label = ku.to_categorical(self.label, num_classes=total_words)

        #return predictors, label, max_sequence_len, total_words

    def generate_text(self,seed_text, next_words):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)
            
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text
#m =myMarkov();
twitter=TwitterAPI();

#twitter.download_images_by_tag("piesciencefiction23", 0, 0, 1, 'dump')
imagelist= glob.glob("dump/*")# choose en image based on key words
print(imagelist)
len1=len(imagelist)
while len(imagelist) ==0:
    twitter.download_images_by_tag("piesciencefiction23", 0, 0, 1, 'dump')
    imagelist= glob.glob("dump/*")# choose en image based on key words
p=sentence_predictor();
i =imagePredictor();
imagelist= glob.glob("dump/*")# choose en image based on key words
imstr=random.choice(imagelist)
predclass = i.predict(imstr)
#res=m.generate()
with open(sys.argv[1]) as f:
    text = f.read()
text_model = markovify.Text(text, state_size=2);
simple_model = markovify.Text(text, state_size=1);
nt=myTextGen()
#res=text_model.make_sentence(tries=100);
res=text_model.make_sentence(tries=100);
res=nt.generate_text(res, 30)
try:
    lastTwoWords = res.split()[-2:]
    print(lastTwoWords)
    laststr=lastTwoWords[0]+" "+lastTwoWords[1]
    print(laststr)
    res=res+ text_model.make_sentence_with_start(laststr,False,tries=100)
except:
    print("failed with last two words")

words=res.split()
print(words)
words.reverse()
revwords=list(words)
print(revwords)
newres=''
if res.endswith('.'):
    newres=res
else:    
    for i,k in zip(revwords[0::2], revwords[1::2]):
        try:
            print(k+" "+i)
            newtest=k+" "+i
            ind=res.find(newtest)
            splitres=res[:ind]
            print(splitres)
            newres=splitres+text_model.make_sentence_with_start(newtest,False,tries=100)
            print(newres)
            break


        except Exception:
            traceback.print_exc()

    res=newres        

res=newres
print(res)

print("new res is")
print(newres)
#while p.predict(res) != predclass or ( len(res)<40 or len(res)>230):
with open(sys.argv[2]) as f:
    text = f.read()
cyberpunk_model = markovify.Text(text, state_size=2);
with open(sys.argv[3]) as f:
    text = f.read()
scifi_model = markovify.Text(text, state_size=2);
with open(sys.argv[4]) as f:
    text = f.read()
fantasy_model = markovify.Text(text, state_size=2);
count=0
try:
    textpred=str(p.predict(res))
except:
    textpred=''
while  textpred != str(predclass) or  ( len(res)<40 or len(res)>230):
    try:
        if count<20:
            res=text_model.make_sentence(tries=100);
            res=nt.generate_text(res, 30)
        else:
            if str(predclass)=='cyberpunk':
                res=cyberpunk_model.make_sentence(tries=100)
                res=nt.generate_text(res, 30)
            elif str(predclass)=='sciencefiction':
                res=scifi_model.make_sentence(tries=100)
                res=nt.generate_text(res, 30)
            elif str(predclass)=='fantasy':
                res=fantasy_model.make_sentence(tries=100)
                res=nt.generate_text(res, 30)
    except:
        continue
    print("generating markov")
    print(res)
    #print(str((p.predict(res))))
    try:
        textpred=str(p.predict(res))
    except:
        textpred=''
    print('predclass '+str(predclass))
    print('textpred '+str(textpred))
    count=count+1
    if count>50 and textpred != predclass:
        count3=0
        while count3<100 and textpred != predclass:
            print("in backup loop")
            if str(predclass)=='cyberpunk':
                res=cyberpunk_model.make_sentence(tries=100)
            elif str(predclass)=='sciencefiction':
                res=scifi_model.make_sentence(tries=100)
            elif str(predclass)=='fantasy':
                res=fantasy_model.make_sentence(tries=100)
            count3=count3+1   
            if count<60 and count3<30:

                res=nt.generate_text(res, 20)
            try:
                textpred=str(p.predict(res))
            except:
                textpred=''
        res=nt.generate_text(res, 30)

    #words=res.split()
    #print(words)
    #words.reverse()
    #revwords=list(words)
    #print(revwords)
    #newres=res
    count2=1
    while count2<20 and len(res)<230 and not (res[-1]=='.' or res[-1]=='?' or res[-1]=='"'):
        words=res.split()
        res=nt.generate_text(res,len(words)+1)
        #res=nt.generate_text(res,count2)
        #words=res.split()
        ind=res.find(words[-1])
        splitres=res[:ind]
        print('splitres '+splitres)
        resbackup=str(res)
        try:
            res=splitres+' '+simple_model.make_sentence_with_start(words[-1],False,tries=100)
        except:
            print("markov error")
            res=resbackup
        
        count2=count2+1
        print(res)

        
    '''
    if res.endswith('.'):
        newres=res
    else:    
        for i,k in zip(revwords[0::2], revwords[1::2]):
            try:
                print(k+" "+i)
                newtest=k+" "+i
                ind=res.find(newtest)
                if ind<int(len(res)/2):
                    continue
                
                splitres=res[:ind]
                print(splitres)
                #newres=splitres+' '+text_model.make_sentence_with_start(newtest,False,tries=100)
                if count>100:
                    newres=res
                    break

                if count<100:
                    #res=text_model.make_sentence(tries=100);
                    #res=nt.generate_text(res, 30)
                    newres=splitres+' '+text_model.make_sentence_with_start(newtest,False,tries=100)
                else:
                    if str(predclass)=='cyberpunk':
                        #res=cyberpunk_model.make_sentence(tries=100)
                        #res=nt.generate_text(res, 30)
                        newres=splitres+' '+cyberpunk_model.make_sentence_with_start(newtest,False,tries=100)
                    elif str(predclass)=='sciencefiction':
                        #res=scifi_model.make_sentence(tries=100)
                        #res=nt.generate_text(res, 30)
                        newres=splitres+' '+scifi_model.make_sentence_with_start(newtest,False,tries=100)
                    elif str(predclass)=='fantasy':
                        #res=fantasy_model.make_sentence(tries=100)
                        #res=nt.generate_text(res, 30)
                        newres=splitres+' '+fantasy_model.make_sentence_with_start(newtest,False,tries=100)
                print(newres)
                break
            except Exception:
                traceback.print_exc()
    '''            
    #res=newres
'''
while ( len(res)<40 or len(res)>230):
    print(p.predict(res)+" "+str(predclass))
    res=text_model.make_sentence(tries=100);
    res=nt.generate_text(res, 7)
    if res.endswith('.') or res.endswith('?'):
        newres=res
    else:    

        try:
            lastTwoWords = res.split()[-2:]
            sys.stderr.write(lastTwoWords)
            laststr=lastTwoWords[0]+" "+lastTwoWords[1]
            sys.stderr.write(laststr)
            res=res+ text_model.make_sentence_with_start(laststr,False,tries=100)
        except:
            sys.stderr.write("failed with last two words")

    words=res.split()
    sys.stderr.write(str(words))
    words.reverse()
    revwords=list(words)
    sys.stderr.write(str(revwords))
    newres=''
    if res.endswith('.'):
        newres=res
    else:    
        for i,k in zip(revwords[0::2], revwords[1::2]):
            try:
                sys.stderr.write(str(k+" "+i))
                newtest=k+" "+i
                ind=res.find(newtest)
                splitres=res[:ind]
                sys.stderr.write(str(splitres))
                newres=splitres+text_model.make_sentence_with_start(newtest,False,tries=100)
                sys.stderr.write(str(newres))
                break


            except Exception:
                traceback.print_exc()

        res=newres        

'''
res=res+"\n#"+str(predclass)
twitter.tweet_media(res,imstr)
'''
print(str("pred class ")+str(predclass))
while p.predict(res) != predclass or ( len(res)<40 or len(res)>230):
    print(p.predict(res))
    print(predclass)
    res=m.generate()
    #res=text_model.make_sentence(tries=100);
    #res=text_model.make_sentence(tries=100);
    res=nt.generate_text(res, 3)
res2=m.generate()
res2=nt.generate_text(res2,3)
while p.predict(res+res2) == predclass and len(res+res2)<220:
    res2=m.generate()
    res2=nt.generate_text(res2,3)
    res=res+res2

print(res) 
#res=res+"\n#"+str(predclass)
#res=res+text_model.make_sentence_with_start(res)
res=res+"\n#"+str(predclass)
twitter.tweet_media(res,imstr)
'''
'''
res=m.generate()
print(p.predict(res))
pred=p.predict(res)
while pred != 'scifi' and len(res)<40:
    res=m.generate()
    pred=p.predict(res)
    print(p.predict_num(res))
print("predicting res :"+res)
print(p.predict_num(res))
#pre=vectorizer.transform([el]).toarray()
#prediction=model.predict_classes([pre])
#m.graph()
print(p.predict(res))
res=m.generate()
print(p.predict(res))
pred=p.predict(res)
while pred != 'cyberpunk' and len(res)< 40:
    res=m.generate()
    pred=p.predict(res)
    print(p.predict_num(res))
print("predicting res :"+res)
print(p.predict_num(res))
#for el in m.tree:
    #print(str(el)+" "+str(m.tree[el]))
#m.graph()
'''
