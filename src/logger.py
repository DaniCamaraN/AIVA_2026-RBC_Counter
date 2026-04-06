import logging

class Logger:

    def __init__(self, ruta_log: str = "rbc_counter.log") -> None:
        logging.basicConfig(
            filename=ruta_log,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def info(self, mensaje: str) -> None:
        logging.info(mensaje)

    def error(self, mensaje: str) -> None:
        logging.error(mensaje)