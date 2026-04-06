import cv2
import numpy as np

class ImageLoader:

    def load_image(self, image_filename: str)->np.ndarray:
        """
        Carga una imagen desde el disco.

        Parameters
        ----------
        image_filename : str
            Ruta del archivo de imagen.

        Returns
        -------
        image : ndarray
            Imagen cargada con OpenCV (BGR).
        """
        image: np.ndarray | None = cv2.imread(image_filename)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_filename}")
        return image