import os

directory = "data/pizza_steak_sushi/train"

if os.path.exists(directory):
    print("Exists")
else:
    print("Doesnt exist")

