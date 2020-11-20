import numpy as np
import sklearn
import skimage
import sklearn.model_selection

#ts and keras
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import EfficientNetB7
from keras.preprocessing.image import ImageDataGenerator

#dimensionality
from sklearn.decomposition import PCA

#classificators
from sklearn.svm import SVC
from sklearn import naive_bayes

#visualization
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#utils
import os
import imageio
from datetime import datetime

classes = ["circular", "falciforme", "outras"]
classificators = ["svm", "bayes"]
batch = 16
epochs = 30
opt = 'adam'
lr = 0.001

def loadDir(k, round): 
  train_dir = './dataset/'+str(k)+'-fold/originais/round_'+str(round)+'/train'
  test_dir = './dataset/'+str(k)+'-fold/originais/round_'+str(round)+'/test'
  return train_dir, test_dir

def createSaveFile(net):
  dir = str(net)+"/"
  if not(os.path.isdir(dir)):
    os.mkdir(dir)

  today = datetime.today()
  path = dir + str(today) + ".txt" 

  saveFile = open(path, 'a')
  return saveFile

def createModel(net):

  if net == "resnet":
    base_model = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
  if net == "eff":
    base_model = EfficientNetB7(weights='imagenet', include_top=True)
  vector = base_model.get_layer("avg_pool").output
  model = tf.keras.Model(base_model.input, vector)

  return model

def toArray(X_list_train, Y_list_train, X_list_test, Y_list_test):
  train_imgs = np.asarray(X_list_train, dtype=np.float32)
  train_labels = np.asarray(Y_list_train, dtype=np.float32)

  test_imgs = np.asarray(X_list_test, dtype=np.float32)
  test_labels = np.asarray(Y_list_test, dtype=np.float32)

  return train_imgs, train_labels, test_imgs, test_labels

def extract_features_test(path, model):
  print('extracting features')
  x_list = []
  y_list = []

  for label in range(3):    
    folder_path = os.path.join(path, classes[label])
    for file in os.listdir(folder_path):    
        file_path = os.path.join(folder_path, file)
        
        if not(file.endswith(".jpg")):
            continue
        
        # load image
        img = image.load_img(file_path, target_size=(224,224))
        # convert image to numpy array
        img_arr = image.img_to_array(img)
        # add 1 more dimension
        img_arr_b = np.expand_dims(img_arr, axis=0)
        # preprocess image
        input_img = preprocess_input(img_arr_b)
        # extract feature
        features = model.predict(input_img)

        x_list.append(features.ravel())
        y_list.append(label)

  return x_list, y_list

def extract_features_train(path, model):
  print('extracting features')
  x_list = []
  y_list = []

  for label in range(3):    
    folder_path = os.path.join(path, classes[label])
    for file in os.listdir(folder_path):    
        file_path = os.path.join(folder_path, file)
        
        if not(file.endswith(".jpg")):
            continue
        
        # load image
        img = image.load_img(file_path, target_size=(224,224))
        # convert image to numpy array
        img_arr = image.img_to_array(img)
        # add 1 more dimension
        img_arr_b = np.expand_dims(img_arr, axis=0)
        # preprocess image
        input_img = preprocess_input(img_arr_b)
        #data augmentation
        da = []
        img_vertical_flip = np.flipud(input_img)
        img_horizontal_flip = np.fliplr(input_img)
        da.append(input_img)
        da.append(img_vertical_flip)
        da.append(img_horizontal_flip)
        # extract feature
        for data in da:
          features = model.predict(data)
          x_list.append(features.ravel())
          y_list.append(label)

  return x_list, y_list

def reduct_features(imgs):
  print('reducting features')
  pca = PCA(n_components=3)
  pca.fit(imgs)
  reduc_features = pca.transform(imgs)
  return reduc_features

def classificate(name, clf, accs, train_features, train_labels, test_features, test_labels, saveFile):
  clf.fit(train_features, train_labels)
  preds = clf.predict(test_features)

  report = classification_report(test_labels, preds, target_names=classes, output_dict=True)
  accs.append(report['accuracy'])
  print("Accuracy: ", report['accuracy'])
  saveFile.write(name + '\n')
  saveFile.write(classification_report(test_labels, preds, target_names=classes))
  return accs

'''
def svm_grid_search(C, kernel, train_X, train_Y):
    accuracy_score_list = []
    
    for c in C:
        # Model training
        svmClassifier = SVC(C = c, kernel = kernel)
        svmClassifier.fit(train_X, train_Y.ravel())
        # Prediction on test set
        pred_y = svmClassifier.predict(train_X)
        # Accuracy
        accuracy = accuracy_score(train_Y, pred_y)
        accuracy_score_list.append(accuracy)
        print('Regularization parameters: ', c, 'Accuracy', accuracy)
    
    max_accurarcy_id = accuracy_score_list.index(max(accuracy_score_list))
    return C[max_accurarcy_id]
'''
net = input("Enter the Net (resnet or eff)")
k = input("Enter the K (must be 5 or 10)")

accsSVM = []
accsBayes = []
model = createModel(net)

saveFile = createSaveFile(net)

for i in range (1, 6):
  
  print("ROUND ", i)

  train_dir, test_dir = loadDir(k, i)

  X_list_train, Y_list_train = extract_features_train(train_dir, model)
  X_list_test, Y_list_test = extract_features_test(test_dir, model)

  train_imgs, train_labels, test_imgs, test_labels = toArray(X_list_train, Y_list_train, X_list_test, Y_list_test)

  print("train/test images shape")
  print(train_imgs.shape)
  print(test_imgs.shape)

  #reduc_features_train = reduct_features(train_imgs)
  #reduc_features_test = reduct_features(test_imgs)

  ##### DESCOMENTE ISSO PARA RODAR SEM PCA 
  reduc_features_train = train_imgs
  reduc_features_test = test_imgs

  print("reduc train/test images shape")
  print(reduc_features_train.shape)
  print(reduc_features_test.shape)

  for clfs in classificators:

    if clfs == "svm":
      clf = SVC(C = 1.0, kernel= 'linear')
      classificate(clfs, clf, accsSVM, reduc_features_train, train_labels, reduc_features_test, test_labels, saveFile)
    if clfs == "bayes":
      clf = naive_bayes.GaussianNB()
      classificate(clfs, clf, accsBayes, reduc_features_train, train_labels, reduc_features_test, test_labels, saveFile)
    

print("Result SVM: ", np.mean(accsSVM))
print("Result Bayes: ", np.mean(accsBayes))
saveFile.write("Result SVM: "+ str(np.mean(accsSVM)))
saveFile.write("Result Bayes: "+ str(np.mean(accsBayes)))

saveFile.close()