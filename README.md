Note corpus has been temporarily removed 

How to use the bot:
Tweet at the bot with the hashtag piesciencefiction23 and it will get an image with that hashtag at when it starts up.
Make sure that all of the .json and .h5 files are in the directory and that there is a folder called dump for storing the images. 

Make sure that the system has the memory to run the bot.

Also remember to use python or pythoon3 depending on your system.

On some systems the wget rule will fail due to firewall rules. If this happens remove all non image files from the dump folder and move your own images into the dump folder.

Dependencies:
Install the following python packages through pip:
keras,
tweepy,
markovify
sklearn,
glob
wget
scipy
numpy
cv2
datetime
nltk
pandas
csv
matplotlib

In the event that another dependency is required it can likely be installed with pip.


To make the system tweet run this command:
python tweetonce.py corpus4.txt cyberpunk3.txt sciencefiction3.txt fantasy2.txt
or potentially:
python3 tweetonce.py corpus4.txt cyberpunk3.txt sciencefiction3.txt fantasy2.txt

this will make the system tweet once

To make the system tweet continuously run:
python tweetcontinuously.py corpus4.txt cyberpunk3.txt sciencefiction3.txt fantasy2.txt
or 
python3 tweetcontinuously.py corpus4.txt cyberpunk3.txt sciencefiction3.txt fantasy2.txt

This will make the bot tweet every 15 minuites

building the text classification model:
run:
python kerastest12.py corpus4.txt cyberpunk3.txt sciencefiction3.txt  fantasy2.txt
or
python3 kerastest12.py corpus4.txt cyberpunk3.txt sciencefiction3.txt  fantasy2.txt

building the image model:
python train_vgg3.py -d newimagedata2/
or
python3 train_vgg3.py -d newimagedata2/


Building the LSTM text generation model:
run:
python3 mymodel5.py corpus4.txt
or
python mymodel5.py corpus4.txt


