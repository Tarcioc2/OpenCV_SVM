import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Press the green button in the gutter to run the script.
def data_pickle():
    dire = './Img'
    categories = ['Cat', 'Dog']
    data = []
    for category in categories:
        path = os.path.join(dire, category)
        label = categories.index(category)
        for img in os.listdir(path):
            makepath = os.path.join(path, img)
            pet_img = cv2.imread(makepath, 0)
            try:
                pet_img = cv2.resize(pet_img, (50, 50))
                image = np.array(pet_img).flatten()

                data.append([image, label])
            except Exception as e:
                pass

    pick_in = open('data1.pickle', 'wb')
    pickle.dump(data, pick_in)
    pick_in.close()


def data_sav():
    pick_in = open('data1.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.98)

    model = SVC(C=1, kernel='poly', gamma='auto')
    model.fit(trainX, trainY)

    pick = open('model.sav', 'wb')
    pickle.dump(model, pick)
    pick.close()


def pred():
    pick_in = open('data1.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.25)

    pick = open('model.sav', 'rb')
    model = pickle.load(pick)
    pick.close()

    prediction = model.predict(testX)
    accuracy = model.score(testX, testY)

    categories = ['Cat', 'Dog']

    print('Accuracy: ', accuracy)
    print('Prediction is: ', categories[prediction[0]])

    petMY = testX[0].reshape(50, 50)
    plt.imshow(petMY, cmap='gray')
    plt.show()


if __name__ == '__main__':
    data_pickle()
    data_sav()
    pred()
