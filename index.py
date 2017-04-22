from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from glob import glob
import os

def main():
	app = ClarifaiApp()
	image_set = create_images('img/',['faces'])
	app.inputs.bulk_create_images(image_set)
	# create training model
	model = app.models.create(model_id='color', concepts=['faces'])
	model.train()

# creates images with concepts from img file
def create_images(img_path, concepts):
  images = []
  for file_path in glob(os.path.join(img_path, '*.jpeg')):
    #print(file_path)
    img = ClImage(filename=file_path, concepts=concepts)
    images.append(img)
  return images


if __name__ == '__main__':
	main()