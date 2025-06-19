import numpy as np

# Provided cameraParams data as a dictionary
CAMERA_PARAMS = {
    'IntrinsicMatrix': np.array([[1.4133e+03, 0, 950.0639],
                                 [0, 1.4188e+03, 543.3796],
                                 [0, 0, 1]]),
    'FocalLength': [1.4133e+03, 1.4188e+03],
    'PrincipalPoint': [950.0639, 543.3796],
    'Skew': 0,
    'RadialDistortion': np.array([-0.0091, 0.0666]),
    'TangentialDistortion': np.array([0, 0]),
    'ImageSize': [1080, 1920], # [height, width]
    # Placeholders for RotationMatrices and TranslationVectors, if needed elsewhere
    'RotationMatrices': np.random.rand(3, 3, 33),
    'TranslationVectors': np.random.rand(33, 3)
}