import face_recognition as fr
import os
import cv2

def classify_face(file_name):
    try: 
        '''
        Load the library images into a dictionary.
        Splitting the dictionary into two lists: values and keys.
        faces_values holds the encryption.
        faces_keys holds the name of the file.
        '''
        faces = get_encoded_faces()
        faces_values = list(faces.values())
        faces_names = list(faces.keys())

        #Reading the file that the user gave. It will be compared to the library file to see who the image is.
        input_faces = cv2.imread(file_name, cv2.IMREAD_COLOR)

        #Finding the face on th image ao it knows where to put the box.
        face_locations = fr.face_locations(input_faces, number_of_times_to_upsample = 3, model='hog' )

        #Encoding all the faces in the image that the user gave.
        unknown_face_encodings = fr.face_encodings(input_faces, face_locations, num_jitters = 50, model = 'small')

        print(f"{unknown_face_encodings}")
        return True

    except Exception as error:
        raise error

def get_encoded_faces():
    try:
        encoded = {}

        for dirpath, dirname, filename in os.walk("./faces"):
            for file in filename : 
                face = fr.load_image_file("faces/" + file)
                encoding = fr.face_encodings(face)[0]
                encoded[file.split(".")[0]] = encoding 
        return(encoded)

    except Exception as error:
        raise error

if __name__ == "__main__":
    try:
        file_name = input("Give me the file name : ")
        classify_face(file_name)

    except Exception as error:
        print(error)

    finally:
        print(":)")