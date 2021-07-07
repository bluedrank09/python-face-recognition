import face_recognition as fr
import os

def classify_face(file_name):
    try: 
        faces = get_encoded_faces()
        print(f"{faces}")
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