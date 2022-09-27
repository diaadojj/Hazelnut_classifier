from argparse import Namespace
import string
from tokenize import Triple
import streamlit as st
import tensorflow as tf
import cv2
import os
import numpy as np
import rdflib
from rdflib import namespace
from rdflib.namespace import RDF, RDFS, OWL, FOAF
import glob




from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Conv2D, MaxPooling2D
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware



import requests
def create_model():
  model = Sequential([
  
  
  Conv2D(32, (3,3), activation='relu', input_shape=(110, 220, 3)
  ),
  MaxPooling2D(2, 2),

  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),

  Conv2D(128, (3,3), activation='relu'),
  MaxPooling2D(2,2),

  Conv2D(128, (3,3), activation='relu'),
  MaxPooling2D(2,2),

  Flatten(),
  Dropout(0.5),

  Dense(512, activation='relu'),
  Dense(17, activation='softmax')
  ])
  model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def hazelnut_type():

  model = create_model()


  batch_size = 100
  test_datagen2 = ImageDataGenerator(rescale=1./255)




  test_generator2 = test_datagen2.flow_from_directory(
          'C:/Users/SAHIN/Desktop/New/Test3', 
          
          target_size=(110, 220),  
          batch_size=batch_size,
          class_mode='categorical')
  
  #intilize classes:


  train_datagen = ImageDataGenerator(rescale=1./255)




  train_generator = train_datagen.flow_from_directory(
          'C:/Users/SAHIN/Desktop/New/classes', 
          target_size=(110, 220), 
          batch_size=batch_size,
          class_mode='categorical') 

 


  labels = (train_generator.class_indices)
  labels = dict((v, k) for k, v in labels.items())







  model.load_weights('C:/Users/SAHIN/Desktop/New/best_model.h5')


  test_generator2.reset()



  pred=model.predict_generator(test_generator2)

  print(pred)



 

  predicted_class_indices = np.argmax(pred,axis=1)



  predictions = [labels[k]
   for k in predicted_class_indices
   ]

  

  print(predictions)
  print(labels)
  print(predicted_class_indices)
  return predictions



def load_image(image_file):
  
  img = Image.open(image_file)
  return img


st.write("""
# Hazelnut  Classification 
Classify hazelnut image using machine learning and Python

""" )
image = Image.open('C:/Users/SAHIN/Desktop/New/logo.png')
st.image(image , caption= 'ML' , use_column_width=True)

st.subheader('Data Information: ')


image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])

if image_file is not None:
  file_details = {"FileName":image_file.name,"FileType":image_file.type}
  st.write(file_details)

  
  files = glob.glob('Test3/imgs/*')
  for f in files:
    os.remove(f)

  img = load_image(image_file)
 
  st.image(img)
  tempDir = "C:/Users/SAHIN\Desktop/New/Test3/imgs"
  imagePath = os.path.join(tempDir,image_file.name)
  print(imagePath)
  with open(os.path.join(tempDir,image_file.name),"wb") as f: 
    f.write(image_file.getbuffer())         
    st.success("Image uploaded")
    prediction = hazelnut_type()
    print(prediction)
    st.write(prediction)

    rdfFile = prediction[0]

    graph = rdflib.Graph()
    graph.parse("C:/Users/SAHIN\Desktop/ha dataset/" + rdfFile + ".rdf")
    output = []

    dd =""

    graph.serialize(format='xml')

    


    for s, p, o in graph:
      #if type(o) == rdflib.term.Literal:
      #st.write(p)
        #output.append(o.toPython())
       # st.write(s)
      st.write(o)
        
    
   #     dd+= "\n"
   #     dd += o.toPython()
        #st.write(( p ))
      #st.write(( s ))
       
    #print .join(output)
   # st.write(( dd ))


    




    
  multiple_files = [('files', (imagePath, open(imagePath, 'rb'), 'image/jpg'))]

  text_data = {'type':prediction}
  headers = {
   # "Authorization" : "xxx",
   'accept': 'application/json',
   'Content-Type': 'application/json'
  }

  r = requests.post("http://localhost:7655/api/image/", files=multiple_files, data=text_data)
 
      


  
 # files = {'file': (os.path.basename(image_file.name), open(image_file.name, 'rb'), 'application/octet-stream'),
 # 'type': prediction }
 # upload_files_url = "url"
 # headers = {'Authorization': access_token, 'JWTAUTH': jwt}
 # r2 = requests.post("http://localhost:44310/api/image/", files=files)



 # your_data = {'files': "C:/Users/SAHIN\Desktop/New/Test/imgs/f.JPG",    'type': prediction
  #  }
  #headers = { 'Content-Type': 'application/json',
  #      'accept': 'application/json',
  #      }

  #st.write(your_data)
  #r = requests.post("http://localhost:7655/api/image/",  headers=headers, data=your_data).json()


  st.write(r)