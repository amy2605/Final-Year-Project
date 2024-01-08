Real-Time Personal Protective Equipment (PPE) Compliance Detection System Using You Only Look Once Version 5

#												                                                                              	#
#			                                 	S	E	T	U	P					                                              #

PS: For easier process use different environment
Software Used: Anaconda, Visual Studio Code, MATLAB


Dataset (the annotation label dataset has already annotated and stored inside "dset" folder. The step here are explained incase there's any additional on the dataset.)
1. Keep all the dataset images inside one folder.
2. In order to rename and change the image format, run the Project.m file on MATLAB.
3. Once all the dataset has been rename and format, the dataset undergoes the annotation process using LabelImg tool.
4. Next, the dataset is randomly sorted and divided into training and validation by running the DivImage.ipynb.


Run YOLOv5 Model
1. Install any necessary requirement inside the Yolov5 folder under requirement.txt file.
2. If the dataset has already change, change the dataset path inside dataset.yml file according to the new dataset image folder for training and validation.
3. Run the TrainingYOLOv5.ipynb


Run PPE Detection System
1. Import the folder inside Visual Studio Code
2. Run the app.py


Video for system demo: https://youtu.be/LctF6mAIaF0
