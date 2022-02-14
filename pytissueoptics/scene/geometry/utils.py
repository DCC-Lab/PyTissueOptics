import numpy as np

from pytissueoptics.scene.geometry import Rotation


def rotateVerticesArray(verticesArray: np.ndarray, r: Rotation) -> np.ndarray:
    rotationMatrix = eulerRotationMatrix(r.xTheta, r.yTheta, r.zTheta)
    return np.einsum('ij, kj->ki', rotationMatrix, verticesArray)


def eulerRotationMatrix(xTheta=0, yTheta=0, zTheta=0) -> np.ndarray:
    rotationMatrix = np.identity(3)
    if xTheta != 0:
        rotationMatrix = np.matmul(_xRotationMatrix(xTheta), rotationMatrix)
    if yTheta != 0:
        rotationMatrix = np.matmul(_yRotationMatrix(yTheta), rotationMatrix)
    if zTheta != 0:
        rotationMatrix = np.matmul(_zRotationMatrix(zTheta), rotationMatrix)
    return rotationMatrix


def _zRotationMatrix(theta) -> np.ndarray:
    cosTheta = np.cos(theta * np.pi / 180)
    sinTheta = np.sin(theta * np.pi / 180)
    return np.asarray([[cosTheta, -sinTheta, 0],
                       [sinTheta, cosTheta, 0],
                       [0, 0, 1]])


def _yRotationMatrix(theta) -> np.ndarray:
    cosTheta = np.cos(theta * np.pi / 180)
    sinTheta = np.sin(theta * np.pi / 180)
    return np.asarray([[cosTheta, 0, sinTheta],
                       [0, 1, 0],
                       [-sinTheta, 0, cosTheta]])


def _xRotationMatrix(theta) -> np.ndarray:
    cosTheta = np.cos(theta * np.pi / 180)
    sinTheta = np.sin(theta * np.pi / 180)
    return np.asarray([[1, 0, 0],
                       [0, cosTheta, -sinTheta],
                       [0, sinTheta, cosTheta]])
