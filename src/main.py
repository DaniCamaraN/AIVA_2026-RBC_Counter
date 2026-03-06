import cv2

def cargar_imagen(image_filename):
    """
    Carga una imagen desde el disco.

    Parameters
    ----------
    image_filename : str
        Ruta del archivo de imagen.

    Returns
    -------
    imagen : ndarray
        Imagen cargada con OpenCV (BGR).
    """
    imagen = cv2.imread(image_filename)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_filename}")
    return imagen