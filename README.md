# 🧊🖥️ crystal.ai
Computer vision algorithm able to detect and classify crystals generated in the presence of certain molecules. The application is based on the PhD work of Bruno Guerreiro, as published [here](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nbyAZasAAAAJ&citation_for_view=nbyAZasAAAAJ:u5HHmVD_uO8C).

## General Architecture
1. **Image Processing and Classification:**
   - The application focuses on crystal microscopy and employs a convolutional neural network (CNN) using TensorFlow Keras for image classification.
   - The `load_sample.py` script loads crystal microscopy images and prepares them for further processing.

2. **Image Processing with borderline2.py:**
   - The `borderline2.py` module is used for image processing, generating new features for the CNN model.

3. **CNN Model Building and Training:**
   - The `image_classification.py` script builds, trains, and evaluates the CNN model for crystal image classification using TensorFlow Keras.
   - Model performance is assessed through classification reports and confusion matrices, providing insights into accuracy.

4. **Flask Web Interface for Model Prediction:**
   - The application includes a Flask web interface allowing users to predict crystal images using the trained CNN model.
   - Users can select a pre-trained model, input a test string, and upload an image for prediction.
   - Flask-WTF is used for form handling, and the interface provides routes for model selection, prediction, and testing.


![part1_traindataset](https://github.com/cryobiochem/crystal.ai/assets/33891979/719d030a-013d-41b6-a45a-2e6cbb4dd615)
![part2_imageprocessing](https://github.com/cryobiochem/crystal.ai/assets/33891979/5d60e1d0-a990-4b5b-a409-899d9e75bfbd)
![part3_detectionboundary](https://github.com/cryobiochem/crystal.ai/assets/33891979/a70513b4-c1c6-45a3-9e37-cee27011b698)
![part4_uploadimage](https://github.com/cryobiochem/crystal.ai/assets/33891979/f7c3771b-cec3-4e76-8416-5964a11a2334)
![part5_prediction](https://github.com/cryobiochem/crystal.ai/assets/33891979/1e444a67-c427-4681-98a0-a11fd5a5d4d3)



---
## `load_sample.py`

### Purpose:
This Python script serves the purpose of loading image samples for crystal microscopy from a specified directory. It utilizes the TensorFlow Keras library for image preprocessing and works with a directory structure where each subfolder represents a different class of images.

### Dependencies:
- `tensorflow.keras.preprocessing.image`: Used for loading and preprocessing images.
- `matplotlib.pyplot`: Used for plotting images.
- `os`: Used for interacting with the operating system.
- `numpy`: Used for numerical operations.

### Function:
```python
load_samples(image_size=(64, 64, 3), data_dir='datasets\\crystal_microscopy')
```

### Parameters:

    `image_size`: Tuple specifying the target size for loaded images (default is (64, 64, 3)).
    `data_dir`: String representing the path to the directory containing the dataset (default is 'datasets\crystal_microscopy').

### Description:

- The script starts by reading images from each subfolder within the specified dataset directory.
- For each subfolder, it loads images, converts them into NumPy arrays, and creates a corresponding label.
- The function returns a dictionary (filename) mapping image indices to their filenames, a dictionary (dic) mapping class indices to class folder names, labels (Y), data (X), and one-hot encoded classes (classes).

### Example Usage:

python
Copy code
import load_sample

filename, dic, Y, X, classes = load_sample.load_samples(image_size)

Note:

- Ensure that you have the required dependencies installed (tensorflow, matplotlib, numpy) before using this script.
- The `data_dir` parameter should point to the directory structure where each subfolder contains images of a specific class.

---
### `load_json_model.py`
## Image Classification using TensorFlow Keras

This Python script demonstrates image classification using a pre-trained deep learning model. It utilizes the TensorFlow Keras library and additional dependencies.

### Dependencies:
- `h5py`: Used for working with HDF5 files.
- `tensorflow.keras.models`: Provides tools for creating and manipulating neural network models.
- `tensorflow.keras.preprocessing.image`: Used for loading and preprocessing images.
- `numpy`: Used for numerical operations.
- `matplotlib.pyplot`: Used for plotting images.
- `joblib`: Used for saving and loading Python objects.
- `from tensorflow.keras.models import load_model`: Used for loading pre-trained models.

### Code Execution:
1. The script imports necessary libraries and dependencies.
2. It attempts to load a pre-trained model (`GPU_image_class_model.h5`) and associated label dictionary (`Label_dic.pkl`) using `load_model` and `joblib.load`, respectively.
3. A sample image (`sample_image.JPG`) is loaded, preprocessed, and displayed using Matplotlib.
4. The loaded model is used to make predictions on the sample image.
5. The predicted class is mapped using the loaded label dictionary.
6. Metrics (assumed to be stored in `metrics.pkl`) are loaded, and a message displaying the predicted class is printed.

### Note:
- The script assumes that the necessary files (`GPU_image_class_model.h5`, `Label_dic.pkl`, `sample_image.JPG`, `metrics.pkl`) are available in the same directory.
- If uncommented, the commented-out code snippet (`#from load_sample import *`) suggests a potential connection to another script (`load_sample.py`) for loading image samples.
- Ensure that all dependencies are installed before running the script.

---
## `borderline.py`

### Purpose:

This Python script performs image processing on a dataset of crystal microscopy images. It utilizes functions from the load_sample module to load the dataset and process each image. The processed images, along with their properties, are saved to individual files.

### Dependencies:

    `load_sample`: Module containing the load_samples function for loading crystal microscopy images.
    `numpy`: Used for numerical operations.
    `pandas`: Used for data manipulation.
    `matplotlib.pyplot`: Used for plotting images.
    `sklearn.preprocessing.normalize`: Used for normalizing images.
    `sys`: Used for printing progress updates.

### Script Flow:

    Import necessary modules and functions.
    Load crystal microscopy images using the load_samples function.
    Define a function image_prop for processing individual images based on specified parameters.
    Loop through all images in the dataset, applying the image_prop function and saving the processed images to files.
    Optionally, concatenate the processed images and save them to an HDF5 file (new_area_data.h5).

### Parameters:

    `resolutions`: Tuple specifying the image dimensions and color channels (default is (512, 512, 3)).
    `filename`, `dic`, `Y`, `X`, `classes`: Loaded from the load_samples function.

### image_prop Function Parameters:

    `numb`: Image index in the dataset.
    `img_shape`: Tuple specifying the image dimensions (rows, columns, color channels).
    `area_threshold`: Float (0 to 1) representing the intensity threshold for area mapping.
    `thicc`: Pixel thickness of the borders.
    `proximity`: Range of proximity for the quadrant check.
   `quadrant_threshold`: Float (0 to 1) representing the threshold for quadrant occupancy.
    `array`: The dataset array.

### Example Usage:

```python
Copy code
from load_sample import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys

resolutions = (512, 512, 3)
filename, dic, Y, X, classes = load_samples(image_size=resolutions)
```

Notes:

- The processed images are saved as PNG files in the 'ML//Test//image_properties//' directory.
- Progress updates are displayed during the image processing loop.
- The script includes commented-out code for concatenating processed images and saving them to an HDF5 file, which can be uncommented and executed separately if needed.

---
## `image_classification02.py`

### Purpose:
This Python script is designed for image classification using a convolutional neural network (CNN). It leverages the TensorFlow Keras library for building and training the model. The dataset is loaded and processed with functions from the `load_sample` and `borderline2` modules.

### Dependencies:
- `load_sample`: Module containing the `load_samples` function for loading crystal microscopy images.
- `borderline2`: Module containing the `ImageProperties` class for image processing.
- `numpy`: Used for numerical operations.
- `pandas`: Used for data manipulation.
- `time`: Used for measuring runtime.
- `tensorflow.keras`: Used for building and training the CNN model.
- `matplotlib.pyplot`: Used for plotting images.
- `sklearn.model_selection.train_test_split`: Used for splitting the dataset into training and testing sets.
- `seaborn`: Used for plotting confusion matrices.
- `sklearn.metrics`: Used for generating classification reports and confusion matrices.

### Script Flow:

1. Import necessary modules and functions.
2. Load crystal microscopy images using the `load_samples` function.
3. Apply image processing using the `borderline2` module.
4. Split the dataset into training and testing sets.
5. Define an `ImageDataGenerator` for data augmentation.
6. Build a CNN model using the TensorFlow Keras Sequential API.
7. Compile and train the model using the training set.
8. Evaluate the model on the testing set and generate classification reports and confusion matrices.

### Parameters:
- `img_shape`: Tuple specifying the image dimensions and color channels (default is (512, 512, 3)).
- `epochs`: Number of epochs for training the model (default is 1).
- `steps_per_epoch`: Number of steps per epoch during training (default is 1).
- `validation_steps`: Number of steps per epoch during validation (default is 1).

### Example Usage:
```python
from load_sample import *
import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.utils import to_categorical

img_shape = (512, 512, 3)
epochs = 1
steps_per_epoch = 1
validation_steps = 1

# ... (rest of the script) ...
```

### Notes:
- The script includes code for training a CNN model, generating classification reports, and plotting confusion matrices.
- Data augmentation is performed using the `ImageDataGenerator`.
- The `ImageProperties` class from `borderline2` is utilized for image processing before model training.
- Model training results, such as metrics and confusion matrices, are visualized using `matplotlib` and `seaborn`.
---
