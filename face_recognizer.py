def classify_face(file_name):
    print(file_name)
    return True


if __name__ == "__main__":
    try:
        file_name = input("Give me the file name : ")
        classify_face(file_name)

    except Exception as error:
        print(error)

    finally:
        print(":)")