import io
import os
import time
import urllib.request
import zipfile

import skimage
import skimage.segmentation

import repype.pipeline
import repype.stage
import repype.status
from repype.typing import (
    Optional,
    PipelineData,
)


DELAY_SECONDS = float(os.environ.get('DELAY', '0'))

def delay() -> None:
    time.sleep(DELAY_SECONDS)


class Download(repype.stage.Stage):
     
    outputs = ['download']

    def process(
            self,
            pipeline: repype.pipeline.Pipeline,
            config: repype.config.Config,
            status: Optional[repype.status.Status] = None,
        ) -> PipelineData:
        url = config['url']
        with urllib.request.urlopen(url) as file:
            data = file.read()
        delay()
        return dict(
            download = data
        )
     

class Unzip(repype.stage.Stage):
     
    inputs   = ['input_id']
    consumes = ['download']
    outputs  = ['image']

    def process(
            self,
            input_id,
            download,
            pipeline: repype.pipeline.Pipeline,
            config: repype.config.Config,
            status: Optional[repype.status.Status] = None,
        ) -> PipelineData:
        contents = zipfile.ZipFile(io.BytesIO(download))
        with contents.open(input_id) as file:
            data = file.read()
        delay()
        return dict(
            image = skimage.io.imread(io.BytesIO(data))
        )
     

class Segmentation(repype.stage.Stage):
     
    inputs  = ['image']
    outputs = ['segmentation']

    def process(
            self,
            image,
            pipeline: repype.pipeline.Pipeline,
            config: repype.config.Config,
            status: Optional[repype.status.Status] = None,
        ) -> PipelineData:
        image = skimage.filters.gaussian(image, sigma = config.get('sigma', 1.))
        threshold = skimage.filters.threshold_otsu(image)
        delay()
        return dict(
            segmentation = skimage.util.img_as_ubyte(image > threshold)
        )
    

class Output(repype.stage.Stage):

    inputs = ['input_id', 'segmentation']

    def process(
            self,
            input_id,
            segmentation,
            pipeline: repype.pipeline.Pipeline,
            config: repype.config.Config,
            status: Optional[repype.status.Status] = None,
        ) -> PipelineData:
        filepath = pipeline.resolve('segmentation', input_id)
        filepath.parent.mkdir(parents = True, exist_ok = True)
        skimage.io.imsave(filepath, segmentation)
        delay()
        return dict()
