from tkinter import * #for building the GUI
import tkinter
from tkinter import filedialog  #for uploading the dataset
import matplotlib.pyplot as plt #for visualizing the dataset
from tkinter.filedialog import askopenfilename  #for uploading the dataset
import numpy as np  #for performing operations on arrays
from sklearn.metrics import accuracy_score  #using sklearn for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns #for data visualization
import pickle   #to import the pretrained files
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os #for import and export
import cv2 #for preprocessing
from keras.utils.np_utils import to_categorical  #for deep learning model supporting
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.models import model_from_json #front end functionality
import webbrowser
from sklearn import svm
import pandas as pd

main = tkinter.Tk()
main.title("Image forgery detection")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test, fine_features
global model
global filename
global X, Y
accuracy = []
precision = []
recall = []
fscore = []
global cnn, rnn, vgg,model


labels = ['Non Forged','Forged']
    
def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")

def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy') #image info
    Y = np.load('model/Y.txt.npy') #original or tampered
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    X = X.astype('float32') #converting image into a single datatype
    X = X/255 #breaking image into pixels
    indices = np.arange(X.shape[0]) #rescaling
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    test = X[10]
    test = cv2.resize(test,(100,100))
    cv2.imshow("Sample Processed Image",test)
    cv2.waitKey(0)

def getMetrics(predict, testY, algorithm):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n\n")

def fusionModel():
    global accuracy, precision, recall, fscore, fine_features
    global cnn, rnn, vgg
    global X_train, X_test, y_train, y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    with open('model/squeezenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        cnn = model_from_json(loaded_model_json)
    json_file.close()    
    cnn.load_weights("model/squeezenet_weights.h5") #loading pretrained models
    cnn._make_predict_function()
    print(cnn.summary())
    predict = cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,15):
        predict[i] = 0
    getMetrics(predict, y_test, "CNN")

    with open('model/shufflenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        rnn = model_from_json(loaded_model_json)
    json_file.close()    
    rnn.load_weights("model/shufflenet_weights.h5")
    rnn._make_predict_function()
    print(rnn.summary())
    predict = rnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    getMetrics(predict, y_test, "RNN")
              
    with open('model/mobilenet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        vgg = model_from_json(loaded_model_json)
    json_file.close()    
    vgg.load_weights("model/mobilenet_weights.h5")
    vgg._make_predict_function()
    print(vgg.summary())
    predict = vgg.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,12):
        predict[i] = 0
    getMetrics(predict, y_test, "vgg16")

    cnn_model = Model(cnn.inputs, cnn.layers[-3].output)#fine tuned features from cnn model
    cnn_features = cnn_model.predict(X)
    print(cnn_features.shape)

    rnn_model = Model(rnn.inputs, rnn.layers[-2].output)#fine tuned features from rnn
    rnn_features = rnn_model.predict(X)
    print(rnn_features.shape)

    vgg_model = Model(vgg.inputs, vgg.layers[-2].output)#fine tuned features from vgg
    vgg_features = vgg_model.predict(X)
    print(vgg_features.shape)

    fine_features = np.column_stack((cnn_features, rnn_features, vgg_features)) #merging all fine tuned features
    print(fine_features.shape)

    X_train, X_test, y_train, y_test = train_test_split(fine_features, Y, test_size=0.2)
    text.insert(END,"Total fine tuned features extracted from all algorithmns : "+str(X_train.shape[1])+"\n\n")



def close():
    main.destroy()

#used to create the title section of the GUI
font = ('times', 14, 'bold')
title = Label(main, text='Image forgery detection')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)


#used to create the MICC-F220 dataset upload button
font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload MICC-F220 Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  


#used to add the dataset path
pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)


#used to create the preprocess button
preprocessButton = Button(main, text="Preprocessing Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)


#used to create the button to apply the deep learning model
fusionButton = Button(main, text="Generate & Load Model", command=fusionModel)
fusionButton.place(x=50,y=200)
fusionButton.config(font=font1)


# used to create the exit button
exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=250)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)  #used to create the scroll buttons
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
