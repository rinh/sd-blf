import sys
from pathlib import Path

import base64
import io

import arrow
import datetime
import time
from io import BytesIO
import requests

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel, Field, create_model
import pillow_avif
from PIL import Image
import numpy as np

import shared


class Avif2JpgRequest(BaseModel):
    url: str = Field(default=False, title="avif image url", description="avif image url")


class RmAnyRequest(BaseModel):
    image: str = Field(default=False, title="Image",
                       description="Image to work on, must be a Base64 string containing the image's data.")
    mask: str = Field(default=False, title="Mask Image",
                      description="Image to work on, must be a Base64 string containing the image's data.")


class RmAnyResponse(BaseModel):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")


def api_middleware(app: FastAPI):
    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if endpoint.startswith('/api'):
            shared.log.info('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        return res


class Api:
    def __init__(self, app: FastAPI):
        self.router = APIRouter()
        self.app = app

        self.add_api_route("/api/v1/hi", self.hi_api, methods=["GET"])
        self.add_api_route("/api/v1/avif2jpg", self.avif2jpg_api, methods=["GET"])
        # self.add_api_route("/api/v1/rm_any", self.rm_any_api, methods=["POST"], response_model=RmAnyResponse)

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def hi_api(self):
        return "hi."

    def avif2jpg_api(self, req: Avif2JpgRequest = Depends()):
        url = req.url

        img = Image.open(BytesIO(requests.get(url).content))
        img_rgb = img.convert('RGB')
        img_bytes = BytesIO()
        img_rgb.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/jpeg")

    # def rm_any_api(self, req: RmAnyRequest):
    #     img = shared.base64_to_img(req.image)
    #     mask = shared.base64_to_img(req.mask)
    #     shared.log.info("rm_any_api start .. ")
    #     from inpaint_anything import lama_inpaint
    #     ia_path = Path(__file__).parent.absolute() / "./depends/inpaint_anything/"
    #     result_img_arr = lama_inpaint.inpaint_img_with_lama(
    #         np.array(img),
    #         np.array(mask),
    #         config_p=str(ia_path / "./lama/configs/prediction/default.yaml"),
    #         ckpt_p=str(ia_path / "./models/big-lama")
    #     )
    #     shared.log.info("rm_any_api end .. ")
    #     return RmAnyResponse(image=shared.img_to_base64(Image.fromarray(result_img_arr)))
