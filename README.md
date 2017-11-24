Train the object-detection model using:
```
python train_phone_finder.py ./data 
```
Test the model using: 
```
python find_phone.py ./data/x.jpg
```
This command will print the normalized coordinates of a phone in the test image and 
show the image with a box around the detected phone.
