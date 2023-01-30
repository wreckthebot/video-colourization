import cv2
import numpy as np


# Load the model
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorization_release_v2.caffemodel')

# Load the video
cap = cv2.VideoCapture('input_video.mp4')

while True:
    # Read the frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reshape the grayscale frame
    input_data = np.zeros((1, 1, 256, 256))
    input_data[0, 0, :, :] = cv2.resize(gray, (256, 256))

    # Run the colourization model
    # out = net.forward_all(data=input_data)

    



    # Get the colourized image
    colourized = out['class8_ab'][0, :, :, :].transpose((1, 2, 0))

    # Convert the colourized image from LAB to BGR
    colourized = cv2.cvtColor(np.uint8(colourized), cv2.COLOR_LAB2BGR)

    # Display the colourized frame
    cv2.imshow('Colourized Frame', colourized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video
cap.release()
cv2.destroyAllWindows()
