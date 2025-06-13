import launch
import sys
import subprocess
import importlib.util

def check_requirements():
    """Check if required packages are available"""
    requirements = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
    ]
    
    missing_packages = []
    
    for module_name, package_name in requirements:
        try:
            if module_name == 'cv2':
                import cv2
            elif module_name == 'PIL':
                from PIL import Image
            elif module_name == 'torch':
                import torch
            elif module_name == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package_name)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    missing = check_requirements()
    
    if missing:
        print(f"Installing missing packages for Frequency Separation Extension: {missing}")
        for package in missing:
            if package == 'opencv-python':
                # Use opencv-python-headless for server environments
                launch.run_pip(f"install opencv-python-headless", "opencv-python-headless for Frequency Separation Extension")
            else:
                launch.run_pip(f"install {package}", f"{package} for Frequency Separation Extension")
    else:
        print("All requirements for Frequency Separation Extension are satisfied.")

if __name__ == "__main__":
    install_requirements()