import base64
from io import BytesIO
import io
import logging
from PIL import Image
import os
from dotenv import load_dotenv

demo = None
load_dotenv()

log = logging.getLogger("sd-blf")
diffusion_api = os.getenv("SD_URL", "http://10.1.0.6:17860")
inpaint_anything_api = os.getenv("INPAINT_ANYTHING_URL", "http://10.1.0.6:27766")


# setup console and file logging
def setup_logging():
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s',
    #                     encoding='utf-8', force=True)
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
        "traceback.border": "black",
        "traceback.border.syntax_error": "black",
        "inspect.value.border": "black",
    }))
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False,
                      suppress=[])
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False,
                     rich_tracebacks=True, log_time_format='%H:%M:%S-%f',
                     level=logging.INFO, console=console)
    rh.set_name(logging.INFO)
    log.setLevel(logging.INFO)
    log.addHandler(rh)


def base64_to_img(base64_string):
    img_bytes = BytesIO(base64.b64decode(base64_string))
    img = Image.open(img_bytes)
    return img


def img_to_base64(img: Image, format='JPEG'):
    _img = img
    buffered = io.BytesIO()
    _img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


setup_logging()
