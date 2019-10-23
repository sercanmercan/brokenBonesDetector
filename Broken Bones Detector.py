
# Importing the Keras libraries and packages
from termcolor import colored
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Train_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('Test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# x_train and y_train
X=training_set.next()
x_train=X[0]
y_train=X[1]

classifier.fit_generator(training_set,
                         samples_per_epoch = 600,
                         nb_epoch = 64,
                         validation_data = test_set,
                         nb_val_samples = 200)

#Predictions
y_pred=classifier.predict(x_train)
#Threshold(num1)        
y_pred=(y_pred >0.9)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
TP = cm[0][0];
TN = cm[1][1];
FN = cm[0][1];
FP = cm[1][0];
Recall= TP/(TP+FN)
Precision= TP/(TP+FP)
F1=(2*Precision*Recall)/(Precision+Recall)
print('Recall: ',Recall,'- Precision: ',Precision,'- F1: ',F1)
#presantation test part                                           
#manual testing to show results 

batch_test_set = test_datagen.flow_from_directory('Presentation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                          class_mode = 'binary')                                        
test_batch_array=batch_test_set.next()
batch_ytest=test_batch_array[1]
predicted=classifier.predict_on_batch(test_batch_array[0])
test_batch_pred=list()
for i in range(len(predicted)):
    if predicted[i]>0.9:
        test_batch_pred.append(1)
    else:
        test_batch_pred.append(0)



print(colored("prediction------real","red"))
for i in range(len(test_batch_pred)):
    if(test_batch_pred[i]==batch_ytest[i]):
        
        if(test_batch_pred[i]==1):
           print(" positive         +")
        if(test_batch_pred[i]==0):
           print(" negative         +")
    else:
         if(test_batch_pred[i]==1):
           print(" positive         -")
         if(test_batch_pred[i]==0):
           print(" negative         -")
        
    



        
        
