
# import image module

from PIL import Image

from PIL import ImageFilter

 

# Open an already existing image

imageObject = Image.open("/home/users/leonardo/App3/seg/images/44.jpg")

 

# Apply edge enhancement filter

edgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE)

 

# Apply increased edge enhancement filter

moreEdgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)

 

# Show original image - before applying edge enhancement filters

#imageObject.show() 

 

# Show image - after applying edge enhancement filter

edgeEnahnced.save("deblur.jpg")

#edgeEnahnced.show()

 

# Show image - after applying increased edge enhancement filter

#moreEdgeEnahnced.show()
moreEdgeEnahnced.save("deblur2.jpg")
