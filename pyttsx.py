import pyttsx3
engine = pyttsx3.init()

opfile = "G:\D&I\YOLO\output.txt"

while True:
    file = open(opfile,"r")
    if file.mode =="r":
        contents = file.read()
        print(contents)
    
    engine.say(contents)
    engine.setProperty('rate',120)  #120 words per minute
    engine.setProperty('volume',0.9)
    engine.runAndWait()
