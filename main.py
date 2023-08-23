import cv2
import matplotlib.pyplot as plt

# HAAR CASCADE FILE
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# LOAD THE IMAGE
image = cv2.imread('test_img1.jpg')
# convert to rgb
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#plt.show()

#resize the image
image= cv2.resize(image, (1400,1000))
import matplotlib.pyplot as plt

# HAAR CASCADE FILE
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# LOAD THE IMAGE
image = cv2.imread('test_img2.jpg')

# CONVERT TO GRAY SCALE IMAGE
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# DETECT FACES
faces = face_cascade.detectMultiScale(gray)

# DRAW BOUNDING BOXES AROUND FACES
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# RESIZE THE IMAGE
image = cv2.resize(image, (1400, 1000))

# CONVERT BGR TO RGB
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# DISPLAY THE IMAGE
plt.imshow(img_rgb)
plt.show()


# convert to gray scale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap= 'gray' )
#plt.show()

# DETECT FACES
faces = face_cascade.detectMultiScale(gray)
print(len(faces))

# display the faces in image
for (x,y,w,z) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+z), (0, 255, 0), 2)
cv2.imshow('Faces', image)
cv2.waitKey(0)
plt.show()