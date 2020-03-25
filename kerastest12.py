import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
lines=[]
#with open(sys.argv[1]) as f:
    #lines = f.readlines()
with open(sys.argv[1]) as f:
    text = f.read()
    lines=re.split(r' *[\.\?!][\'"\)\]]* *', text)+f.readlines()
security=[]
tech=[]
fantasy=[]
'''
with open(sys.argv[2]) as f:
    security = f.readlines()
with open(sys.argv[3]) as f:
    tech = f.readlines()
with open(sys.argv[3]) as f:
    fantasy = f.readlines()
'''    
with open(sys.argv[2]) as f:
    text = f.read()
    security= re.split(r' *[\.\?!][\'"\)\]]* *', text)+f.readlines()
with open(sys.argv[3]) as f:
    text = f.read()
    tech=re.split(r' *[\.\?!][\'"\)\]]* *', text)+f.readlines()
with open(sys.argv[4]) as f:
    text = f.read()
    fantasy = re.split(r' *[\.\?!][\'"\)\]]* *', text)+f.readlines()
print(security)
print(tech)
regex = re.compile('[^a-zA-Z ]')
lines = [regex.sub('',w) for w in lines]
security = [regex.sub('',w) for w in security]
tech = [regex.sub('',w) for w in tech]
fantasy = [regex.sub('',w) for w in fantasy]
vectorizer = CountVectorizer(ngram_range=(0,2),min_df=0, lowercase=False)
vectorizer.fit(lines)
print(vectorizer.vocabulary_)
#exit()
train=[]
labels=[]
trainsec=vectorizer.transform(security).toarray()
print("train sec is ")
print(trainsec)
for el in security:
    labels.append('cyberpunk')
traintech=vectorizer.transform(tech).toarray()
print((vectorizer.transform(tech))[0])
#exit()
print("train sec is ")
print(traintech)

for el in tech:
    labels.append('sciencefiction')

for el in fantasy:
    labels.append('fantasy')
trainfantasy=vectorizer.transform(fantasy).toarray()
train=[]
print(tech)
#exit()
train=security+tech+fantasy
train=vectorizer.transform(train).toarray()
#train.append(trainsec)
#trainsec.append(traintech)
#train=numpy.concatenate(trainsec,traintech)
#labels=np.array(labels)
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
print(labels)
np.save('textclasses.npy',le.classes_)

model = Sequential()
model.add(Dense(230, input_dim=len(train[0]), activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))
# compile the keras model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
# fit the keras model on the dataset
#model.fit(train, labels, epochs=30, batch_size=100, verbose=1)
model.fit(train, labels, epochs=50, batch_size=150, verbose=1)
#model.fit(train, labels, epochs=5, batch_size=100, verbose=1)

for el in security:
    print(vectorizer.transform([el]).toarray())
    pre=vectorizer.transform([el]).toarray()
    prediction=model.predict_classes([pre])
    print(prediction)
    print(le.inverse_transform(prediction))
    #print(le.inverse_transform(prediction[0]))
for el in tech:
    print(vectorizer.transform([el]).toarray())
    pre=vectorizer.transform([el]).toarray()
    prediction=model.predict_classes([pre])
    print(le.inverse_transform(prediction))
    #print(le.inverse_transform(prediction[0]))
for el in fantasy:
    print(vectorizer.transform([el]).toarray())
    pre=vectorizer.transform([el]).toarray()
    prediction=model.predict_classes([pre])
    print(le.inverse_transform(prediction))
    #print(le.inverse_transform(prediction[0]))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")        
for el in tech:
    print(vectorizer.transform([el]).toarray())
    pre=vectorizer.transform([el]).toarray()
    prediction=model.predict_classes([pre])
    print(le.inverse_transform(prediction))
