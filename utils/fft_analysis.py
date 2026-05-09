import cv2
import numpy as np
import matplotlib.pyplot as plt

class FFTAnalyzer:
    """
    Analyzes images in the frequency domain to identify deepfake artifacts.
    """
    @staticmethod
    def get_spectrum(image):
        """
        Computes the 2D Fast Fourier Transform and returns the magnitude spectrum.
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Compute FFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        
        # Magnitude spectrum
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        
        # Normalize to 0-255 for display
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return magnitude_spectrum

    @staticmethod
    def save_spectrum_plot(image, save_path):
        """
        Generates and saves a visual plot of the FFT spectrum with a colormap.
        """
        spectrum = FFTAnalyzer.get_spectrum(image)
        # Apply colormap for better visualization
        colored_spectrum = cv2.applyColorMap(spectrum, cv2.COLORMAP_MAGMA)
        cv2.imwrite(save_path, colored_spectrum)
        return save_path

if __name__ == "__main__":
    print("FFTAnalyzer module loaded.")
