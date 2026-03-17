from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import json
import cv2
import numpy as np
from common_code.tasks.service import get_extension

api_description = """
This service resizes an image. It can be used to resize an image to a specific width and height,
or to resize an image by a percentage.
the settings parameter is a JSON object with the following fields:
- width: the width of the resized image (eg. 500)
- height: the height of the resized image (eg. 500)
- scale_percent: the percentage of the original image size (eg. 50)
- with_ratio: if true, the width and height fields are ignored and the image is resized by the scale_percent field
"""
api_summary = """
Resizes an image.
"""

api_title = "Image Resize API."
version = "0.0.1"

settings = get_settings()


class MyService(Service):
    """
    Image resize model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Image Resize",
            slug="image-resize",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
                FieldDescription(name="settings", type=[FieldDescriptionType.APPLICATION_JSON])
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING
                ),
            ],
            has_ai=False,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/image-resize/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        raw = data["image"].data
        input_type = data["image"].type
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
        raw_settings = data["settings"].data
        image_settings = json.loads(raw_settings)

        if "with_ratio" in image_settings and image_settings["with_ratio"] is True:
            scale_percent = image_settings["scale_percent"]
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
        else:
            width = image_settings["width"]

            if "height" in image_settings:
                height = image_settings["height"]
            else:
                ratio = width / img.shape[1]
                height = int(img.shape[0] * ratio)

        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Save .jpg image
        guessed_extension = get_extension(input_type)
        is_success, out_buff = cv2.imencode(guessed_extension, resized)

        return {
            "result": TaskData(
                data=out_buff.tobytes(),
                type=input_type
            )
        }
