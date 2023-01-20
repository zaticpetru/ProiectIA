import face_recognition as fr
import os
import cv2
import matplotlib.pyplot as plt
import face_recognition
import numpy as np
from time import sleep
from PIL import Image
import shutil

def get_encoded_faces(faces_available):
    """
    cauta in folder si incripteaza
    toate fetele

    :return: dict of (filename, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk(faces_available):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):

                face = fr.load_image_file(faces_available + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f] = encoding

    return encoded


# def unknown_image_encoded(img):
#     """
#     incripteaza o imagine din folder
#     """
#     face = fr.load_image_file("faces/" + img)
#     encoding = fr.face_encodings(face)[0]

#     return encoding


def classify_face(im, faces_available):
    """
    va gasi toate fetele si le va clasifica daca
    le recunoaste din baza de date

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces(faces_available)
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # cauta daca fata face parte din fetele recunoscute
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # !foloseste fetele cu cea mai apropiata distanta 
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # !foloseste toate fetele recunoscute 
        # for i, match in enumerate(matches):
        #     if(match):
        #         face_names.append(known_face_names[i])

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Deseneaza marginile fetei
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Deseneaza un lable cu numele persoanei
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Arata imaginea rezultata
    # while True:
        # print(img)
        # plt.imshow(img)
        
        # plt.show()
        # im = Image.open(img)
        # im.show()

        # if plt.waitforbuttonpress(1):
    return face_names 


# results = classify_face("./testImages/test1.jpg", "./faces/")

# counter = 0
# for filename in results:
#     counter += 1
#     shutil.copyfile("./faces/" + filename, "./result/" + str(counter) + " " + filename)
# print(results)

results = classify_face("./testImages/Keanu Reeves.jpg", "./faces2/")
# results.extend(classify_face("./testImages/Keanu Reeves.jpg", "./faces2/"))
counter = 0
for filename in results:
    counter += 1
    shutil.copyfile("./faces2/" + filename, "./result2/" + str(counter) + " " + filename)
print(results)

# results = classify_face("./testImages/Leonardo Di Caprio.jpg", "./faces2/")
# # results.extend(classify_face("./testImages/Keanu Reeves.jpg", "./faces2/"))
# counter = 0
# for filename in results:
#     counter += 1
#     shutil.copyfile("./faces2/" + filename, "./result2/" + str(counter) + " " + filename)
# print(results)