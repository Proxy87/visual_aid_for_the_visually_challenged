# visual_aid_for_the_visually_challenged
This project intends to identify the objects in the surrounding of the person holding the camera and give an audio output  of the detected objects. This project is done using openCV and python.

There are two parts to this project, the first script yolo.py, uses yolo algorithm to detect objects from the frames which are obtained from the camera feed. The description of the detected objects(name, object location i.e either left,right or at the center of the frame,) are saved in a .txt file in the given path. The second script pysstx.py reads this .txt file and gives audio output of the first line in the file.
