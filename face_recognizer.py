import sys
import face_recognition as fr
import numpy as np
import os
import cv2
import logging as log
import inspect

def classify_face(file_name):
    try: 
        log.debug(f"In {inspect.stack()[0][3]}")
        '''
        Load the library images into a dictionary.
        Splitting the dictionary into two lists: values and keys.
        faces_values holds the encryption.
        faces_keys holds the name of the file.
        '''
        log.info(f"Starting to encode faces in the file given.")
        faces = get_encoded_faces()
        faces_values = list(faces.values())
        faces_names = list(faces.keys())
        log.info(f"There are {len(faces_names)} names in the faces library.")
        
        #Reading the file that the user gave. It will be compared to the library file to see who the image is.
        log.info(f"Reading the file you have given.")
        input_faces = cv2.imread(file_name)
        
        #Finding the face on the image so it knows where to put the box.
        log.info(f"Finding all the faces in the file given.")
        face_locations = fr.face_locations(input_faces, number_of_times_to_upsample = 3, model='hog' )
        log.debug(f"Face locations are : {face_locations}")
        log.info(f"Found {len(face_locations)} faces in the file given.")

        #Encoding all the faces in the image that the user gave.
        log.info(f"Encoding the unknown faces in the file.")
        unknown_face_encodings = fr.face_encodings(input_faces, face_locations, model = 'small')
        log.debug(f"Unknown faces are : {unknown_face_encodings}")

        log.info(f"Starting face comparisons")

        face_names = []

        for list_index, face_encoding in enumerate(unknown_face_encodings):
            # See which picture in the library is the best match with the encoded face(s).
            log.info(f"Getting all the good matches for each face from the given picture with the faces in the faces library.")
            matches = fr.compare_faces(faces_values, face_encoding)
            name = "Unknown" 
            log.info(f"Comparing face {list_index + 1}")
            log.info(f"Got all the good matches found for the faces in the file given ")
            face_distances = fr.face_distance(faces_values, face_encoding)
            best_match_index = np.argmin(face_distances)
            log.debug(f"Best match index : {best_match_index}, Face names : {faces_names}")

            if matches[best_match_index]:
                name = faces_names[best_match_index]

            log.info(f"Got {name} as the best match out of all the good matches for the faces in the given file.")

            #Removing any additional characters.
            #The files are called the name of the person and a number at the end. This removes the number and the space.
            original_name = len(name)
            new_name = name[:original_name-2]
            log.debug(f"{inspect.stack()[0][3]} : Changed name")
            log.info(f"Changed the name so that any additional characters are removed.")

            #Getting the best match for the image given from the faces library. This name will show up on screen.
            face_names.append(new_name)
            log.debug(f"{inspect.stack()[0][3]} : Face names list = {face_names}")
            log.info(f"Got the name of the best match. This is the name that will show up on screen.")

            if name != 'Unknown':
                log.info(f"Drawing a box and label around the faces in the file given.")

                for(top, right, bottom, left), name in zip(face_locations, face_names):
                    #Drawing the box around the face.
                    cv2.rectangle(input_faces, (left-20, top-20), (right+20, bottom+20), (106, 13, 137), 2)

                    #Drawing the label for the face in the box
                    cv2.rectangle(input_faces, (left-20, bottom-10), (right+20, bottom+20), (123, 104, 238), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(input_faces, name, (left-20, bottom+15), font, 0.5, (50,205,50), 1)

            else:
                print(f"Face not found :(")

        log.debug(f"High school faces : {get_high_school_faces(face_names, input_faces)}")

        #Creating the window name for the images
        log.info(f"Creating the window names for each window that will load.")
        
        for high_school_name, image_displaying in get_high_school_faces(face_names, input_faces).items():
            cv2.imshow(high_school_name.split('.')[0], image_displaying)

        log.info(f"Showing the windows now. They should appear in your task bar.")
  
        while True:
            if cv2.waitKey(0):
                return(face_names)

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        raise error

def get_encoded_faces():
    try:
        encoded = {}

        log.info(f"Starting to encode the high school photos.")

        #Encoding all the high school faces
        for dirpath, dirname, filename in os.walk("./faces"):
            for file in filename : 
                face = fr.load_image_file("faces/" + file)
                log.debug(f"{inspect.stack()[0][3]} : {file} : Face = {fr.face_encodings(face)}")
                encoding = fr.face_encodings(face)[0]
                #Getting rid of the file extention : .jpg, .png, etc
                encoded[file.split(".")[0]] = encoding 

        log.info(f"Finished encoding all the faces in the faces library")
        
        return(encoded)

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        raise error

def get_high_school_faces(face_names, input_faces):
    try:
        log.info(f"Getting high school faces")

        #Getting all the original faces names to compare with the names of the high school photos. Adding it to a list.
        high_school_names = {'Your file' : input_faces}

        for dirpath, dirname, filename in os.walk("./high school faces"):
            for file in filename:
                log.debug(f"File : {file}, dirpath : {dirpath}, dirname : {dirname}")

                #Checking to see if the high school picture file name (without extention) is also in the original faces file. 
                #If it is, it will be appended to the high school names list.
                log.info(f"Checking that the high school file name adds up with the names of the original faces files.")

                if file.split('.')[0] in face_names:
                    high_school_image = cv2.imread(os.path.join(dirpath, file), cv2.IMREAD_COLOR)
                    high_school_names.update({file : high_school_image})
        
        log.info(f"Finished getting all the names of the high school faces file")

        return high_school_names
                    
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        raise error

if __name__ == "__main__":
    try:
        #Setting log configuration
        log.basicConfig(format = '%(asctime)s : %(lineno)d : %(message)s', level = log.INFO)
        log.info(f"Starting face rec program")
        
        file_name = input("Give me the file name : ")
        log.debug(f"{inspect.stack()[0][3]} Input file name is {file_name}")
        classify_face(file_name)

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.info(f"{inspect.stack()[0][3]} : {exc_tb.tb_lineno} : {error}")
        
    finally:
        print(f":)")