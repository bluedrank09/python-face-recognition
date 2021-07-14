import sys
import face_recognition as fr
import numpy as np
import os
import cv2
import logging as log
import inspect

def classify_face(file_name):
    try: 
        log.info(f"In {inspect.stack()[0][3]}")
        '''
        Load the library images into a dictionary.
        Splitting the dictionary into two lists: values and keys.
        faces_values holds the encryption.
        faces_keys holds the name of the file.
        '''
        faces = get_encoded_faces()
        faces_values = list(faces.values())
        faces_names = list(faces.keys())
        log.info(f"{inspect.stack()[0][3]} : {len(faces_names)} names in library are {faces_names}")
        
        #Reading the file that the user gave. It will be compared to the library file to see who the image is.
        input_faces = cv2.imread(file_name, cv2.IMREAD_COLOR)

        #Finding the face on the image so it knows where to put the box.
        face_locations = fr.face_locations(input_faces, number_of_times_to_upsample = 3, model='hog' )

        #Encoding all the faces in the image that the user gave.
        unknown_face_encodings = fr.face_encodings(input_faces, face_locations, num_jitters = 50, model = 'small')

        log.debug(f"Starting face comparisons")
        face_names = []
        for face_encoding in unknown_face_encodings:
            # See which picture in the library is the best match with the encoded face(s).
            matches = fr.compare_faces(faces_values, face_encoding)
            name = "Unknown" 
        log.debug("Finished face comparisons")
        log.info(f"Matches : {matches}")

        face_distances = fr.face_distance(faces_values, face_encoding)
        best_match_index = np.argmin(face_distances)
        log.debug(f"Best match index : {best_match_index}, Face names : {faces_names}")
        if matches[best_match_index]:
            name = faces_names[best_match_index]
            log.info(f"{inspect.stack()[0][3]} : Name = {name}")

        face_names.append(name)
        log.info(f"{inspect.stack()[0][3]} : Face names list = {face_names}")

        for(top, right, bottom, left), name in zip(face_locations, face_names):
            #Drawing the box around the face.
            cv2.rectangle(input_faces, (left-20, top-20), (right+20, bottom+20), (106, 13, 137), 2)

            #Drawing the label for the face in the box
            cv2.rectangle(input_faces, (left-20, bottom-15), (right+20, bottom+20), (123, 104, 238), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(input_faces, name, (left-20, bottom+15), font, 1.0, (50,205,50), 1)

        while True:
            cv2.imshow('Video', input_faces)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return(face_names)

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        raise error

def get_encoded_faces():
    try:
        encoded = {}

        for dirpath, dirname, filename in os.walk("./faces"):
            for file in filename : 
                face = fr.load_image_file("faces/" + file)
                log.debug(f"{inspect.stack()[0][3]} : {file} : Face = {fr.face_encodings(face)}")
                encoding = fr.face_encodings(face)[0]
                encoded[file.split(".")[0]] = encoding 
        return(encoded)

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        raise error

if __name__ == "__main__":
    try:
        log.basicConfig(format = '%(asctime)s : %(lineno)d : %(message)s', level = log.DEBUG)
        log.info(f"{inspect.stack()[0][3]} Starting face rec program")
        
        file_name = input("Give me the file name : ")
        log.debug(f"{inspect.stack()[0][3]} Input file name is {file_name}")
        print(classify_face(file_name))

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        

    finally:
        print(":)")