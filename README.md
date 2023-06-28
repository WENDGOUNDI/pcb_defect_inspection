# PCB DEFECT INSPECTION

# OVERVIEW
The scope of this project is to develop a defect inspection model using an object detector network. The Pretrained Yolo-Nas large model has been used as the baseline model. The dataset consists to PBC images with proper and defective resistors. The final product will be to leverage a webapp for PBC inspection. 

# DEPENDENCIES
 - Super-Gradients
 - Glob
 - Pandas
 - Streamlit
 - Pillow
 - Collections

# DATASET
The dataset is from Google Cloud Skill Boost course titled <a href="https://www.cloudskillsboost.google/focuses/34180?parent=catalog">Create a Component Anomaly Detection Model using Visual Inspection AI</a>. The downloaded dataset is large of 998 images where 99% were positive images. Therefore we created a subset of the original dataset, composed of 300 images where we followed the original dataset format to create defective components. In the original dataset, defective resistors where defined with a 2d rectangle on their top. We reproduced the same strategy by using 2d shapes (capsule, circle square and rounded square) from Microsoft 3D Paint software to create our defective resistors. For generalization, we used different 2d shapes as mentioned earlier and filled with different colors.

# TRAINING
The inspection model is based on a pretrained yolo-nas large model trained for 500 epochs.

To train the model, open **google_defect_pbc_yolonas.py**, adjust the parameters according to your needs and run the command `python google_defect_pbc_yolonas.py` .

To test, adjust the testing image, the output folder to save predictions and custom model paths and then run the command `python test_custom_model.py` .

To launch the webapp, run `streamlit run streamlit_app_defect_inspection.py` .

# INFERENCE
![result_1](https://github.com/WENDGOUNDI/pcb_defect_inspection/assets/48753146/66735464-ae64-4bd6-b80b-23743f0570aa)

![result_2](https://github.com/WENDGOUNDI/pcb_defect_inspection/assets/48753146/0e9eeace-c153-41ba-8680-0448105caa92)

![result_3](https://github.com/WENDGOUNDI/pcb_defect_inspection/assets/48753146/a0d374c2-6dff-4db3-abe0-6b1973f79ac6)

## WebApp

For better visualization, we created a webpage allowing the user to upload its picture for inspection. Images with the following extensions are supported: JPG, JPEG, PNG, BMP and WEBP. We output a statistical histogram showing the ratio betwwen proper and defective resistor. A slide bar is available for the user to adjust the confidence score whom the default value is 50.

![streamlit](https://github.com/WENDGOUNDI/pcb_defect_inspection/assets/48753146/b8bd9868-e07e-4ef9-af2c-88a7b2dcd37f)

# Notes
 - The use of the dataset is for education purpose  only and the dataset belongs to Google
 - Here is my recognition <a href="https://www.cloudskillsboost.google/public_profiles/b7ea4169-1267-4a22-8cf1-0fc6de36ba8d/badges/4057685">badge text</a> for successfully completing the training and passed the exam.



