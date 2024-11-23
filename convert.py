import os,sys

path = "."


for file in os.listdir(path):
    #    if file.endswith("HEIC"):
#        print(f"HEIC: {file}")
#    else:
#        print(f"JPEG: {file}")
    file_path = file.rsplit('.',1)[1]
    print(file_path)

