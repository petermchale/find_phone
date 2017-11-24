# Object-detection algorithm
Python 2.x code to compute the 
location of a phone on a floor from a single RGB camera image. 
<img src='data/detected_phone.png' width='600'>
## Usage 

Train the object-detection model using:
```
python train_phone_finder.py ./data 
```
The data set consists of approximately 100 jpeg images of the 
floor from a factory building with a phone on it. 
There is a file named `labels.txt` that contains 
normalized coordinates of a phone for each picture.

Test the model using: 
```
python find_phone.py ./data/x.jpg
```
which will print to the terminal 
the normalized coordinates of the phone in the test image and 
show the image with a box around the detected phone.
