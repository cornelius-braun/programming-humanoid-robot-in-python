"""
helpers to marshall and unmarshall the 4,4 transforms
"""
import numpy as np

def marshall(transformation: np.array):
    return [n.item() for n in transformation.flatten()]

def unmarshall(numbers):
    return np.array(numbers).reshape(4, 4)