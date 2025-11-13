import logging
import os


def create_log_id(dir: str) -> int:
    """Create a unique log ID for the given directory."""
    log_count = 0
    file_path = os.path.join(dir, f"log{log_count:d}.log")
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir, f"log{log_count:d}.log")
    return log_count


def logging_config(
    folder: str,
    name: str,
    lavel=logging.DEBUG,
    console_level=logging.DEBUG,
    no_console: bool = True,
):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print(f"All logs will be saved to {logpath}")

    logging.root.setLevel(lavel)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(lavel)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(formatter)
        logging.root.addHandler(console)

    return folder
