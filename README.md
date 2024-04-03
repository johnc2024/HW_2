**Introduction**

This project leverages machine learning algorithms for image classification to identify COVID-positive individuals from a collection of chest X-ray scans (n=4,035) taken from individuals with no respiratory condition, individuals infected with viral pneumonia, and individuals infected with COVID-19. The analysis is conducted using a combination of convolutional neural networks (CNN) and transfer learning models implemented via the TensorFlow and Keras frameworks in Python. 

**Data processing**

A total of 1,345 X-rays from each response category are drawn from the source data, set to RGB format, and resized to either 192 by 192 pixels or 224 by 224 pixels depending on the classification method applied. Data augmentation steps (i.e. random flips and random rotations of X-ray images) are executed concurrently with model training to address overfitting as needed.

**Exploratory analysis**

Structural similarity index measures (SSIM) are calculated on randomly selected pairs of COVID and pneumonia X-rays (0.492), COVID and normal X-rays (0.326), and pneumonia and normal X-rays (0.359), for cross-category image comparison. 

**Image classification**

__Model training__

Two custom CNN models are initially trained on the X-ray image data over five epochs, each comprised of twelve 2D-convolution layers, five max-pooling layers, and a softmax activation layer — one with Adam optimization and a batch size of 64, and another with stochastic gradient descent (SGD) optimization and a batch size of 128. The latter CNN model additionally includes dropout layers to further minimize overfitting.

Two transfer learning models, VGG19 and ResNet50 (pre-trained on ImageNet data), are subsequently fitted to the X-ray image data over five epochs. Both models rely on Adam optimization, and are fine-tuned with base layers unfrozen before being refitted to the X-ray image data.

Finally, a sparse CNN model with only five 2D-convolution layers, four max-pooling layers, and a softmax activation layer is trained over ten epochs on the X-ray image data.

__Results__

The sparse CNN model exhibits the highest performance among the five candidate models, with training and validation accuracies of 0.760 and 0.794, respectively. The classification performance reaches only 0.333, an outcome potentially attributable to overfitting and limited generalizability to external data.

**Sources**

Géron, A. (2019), Hands-on machine learning with SCIKIT-learn, Keras, and tensorflow: Concepts, tools, and Techniques, O’Reilly Media

J. Deng, W. Dong, R. Socher, L. -J. Li, Kai Li and Li Fei-Fei, "ImageNet: A large-scale hierarchical image database," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 2009, pp. 248-255, https://ieeexplore.ieee.org/document/5206848

Keras Documentation: Resnet and RESNETV2, accessed April 2, 2024, https://keras.io/api/applications/resnet/ 

Keras Documentation: Transfer Learning & Fine-tuning, accessed April 2, 2024, https://keras.io/guides/transfer_learning/

Keras Documentation: VGG16 and VGG19, accessed April 2, 2024, https://keras.io/api/applications/vgg/ 

M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, “Can AI help in screening Viral and COVID-19 pneumonia?” arXiv preprint, 29 March 2020, https://arxiv.org/abs/2003.13145

Transfer learning & fine-tuning: Tensorflow Core, TensorFlow, accessed April 2, 2024, https://www.tensorflow.org/guide/keras/transfer_learning 
