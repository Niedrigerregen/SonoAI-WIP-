*SonoAI*
A TensorFlow and Keras-based model for detecting fetal brain tumors in ultrasound images.

----

Overview:

SonoAI is an experimental deep learning project exploring whether Convolutional Neural Networks (CNNs) can effectively learn from extremely small and unevenly distributed datasets in a highly complex medical imaging domain — fetal brain tumor detection.

This project aims to push the limits of CNN generalization and augmentation strategies when data scarcity and imbalance are major challenges.

----

Project Goals:

	* Investigate the performance of CNNs on tiny, imbalanced medical datasets.
	* Explore data augmentation and regularization techniques to improve generalization.
	* Evaluate how model architecture depth and callback strategies affect training stability.
	* Assess whether meaningful predictions can still emerge from limited, noisy ultrasound data.

----

 Model Details:
 
	* Frameworks: TensorFlow & Keras
	* Core Model: Custom CNN architecture with multiple convolutional and pooling layers
	* Techniques Used:

		* Early Stopping
		* Data augmentation (rotation, zoom, flip, etc.)stopping and learning rate scheduling
		* Dropout and batch normalization
		* Custom callbacks for model monitoring

----

Repository Structure:

SonoAI/

│

├── SonoAI_Code.py                # First Version of the CNN (bad)

├── SonoAI_Code - Kopie.py        # Newest/Best Version yet (i copied the script from another folder that's why it's got "Kopie" behind it)

----

Results & Observations:

	* The model demonstrates partial learning capability even with minimal data.
	* Heavy augmentation and careful tuning are essential to avoid overfitting.

----

Future Work:

	* Expand dataset size and improve labeling accuracy.
	* Experiment with transfer learning using pre-trained medical imaging models.
	* Integrate Grad-CAM or similar visualization tools for interpretability.
	* Evaluate model performance using cross-validation and external test sets.

----

Disclaimer:

I did not create the dataset required for the CNN

This project is for research and educational purposes only.

It is not intended for clinical or diagnostic use.

----
Dataset Attribution:

This project uses the Ultrasound Fetus Dataset available on [Kaggle](https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset).
The dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to share and adapt the material for any purpose, even commercially, as long as appropriate credit is given.

Citation
Anitha, A (2024). Ultrasound Fetus Dataset. Mendeley Data, V1, doi: 10.17632/yrzzw9m6kk.1

----

Author:
 
Developed by Loran Hisso
Exploring the boundaries of deep learning with limited medical data.
