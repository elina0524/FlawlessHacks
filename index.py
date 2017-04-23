from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from glob import glob
from PIL import Image
import os


def main():
	app = ClarifaiApp()

	# get image height and width
	img = Image.open('./img/face1.jpeg')
	width, height = img.size

	# detect boundaries of face
	face_model = app.models.get('face-v1.3')
	image = ClImage(file_obj=open('./img/face1.jpeg', 'rb'))
	face_result = face_model.predict([image])
	#print(face_result)
	face = face_result['outputs'][0]['data']
	
	# get boundaries for face
	for i in face.values():
		for j in i:
			top = i[0]['region_info']['bounding_box']['top_row']
			left = i[0]['region_info']['bounding_box']['left_col']
			right = i[0]['region_info']['bounding_box']['right_col']		
			bottom = i[0]['region_info']['bounding_box']['bottom_row']
	# calculate height and width
	top = int(top*height)
	left = int(left*width)
	right = int(right*width)
	bottom = int(bottom*height)

	# crop image
	img2 = img.crop((left,top,right,bottom))
	img2 = img2.resize((width, height)).save('./img/new.jpeg')
	# detect color in cropped image
	image2 = ClImage(file_obj=open('./img/new.jpeg', 'rb'))
	color_model = app.models.get('color')
	color_result = color_model.predict([image2])
	#print(result)
	
	color = color_result['outputs'][0]['data']
	#print(color)
	a = 0
	mval = 0
	for i in color.values():
		for j in i:
			hex_dec = i[a]['w3c']['hex']
			name = i[a]['w3c']['name']
			value = i[a]['value']
			a = a+1
			if hex_dec != '#ffffff' or hex_dec != '#000000':
				if value > mval:
					mval = value
					skin = name
					dec = hex_dec
		print(mval, skin, dec)

	#print(hex_dec, name)	

if __name__ == '__main__':
	main()

