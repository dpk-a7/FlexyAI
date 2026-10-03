"""The explicit server-side allowlist for the currently supported native apps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AppInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    prompt: str = Field(min_length=1)
    seed: int = 1


class MiniMaxI2VInput(AppInput):
    duration_seconds: float = Field(default=5, gt=0)
    megapixels: float = Field(default=0.9, gt=0)


class MiniMaxR2VInput(AppInput):
    duration_seconds: float = Field(default=10, gt=0)
    megapixels: float = Field(default=0.9, gt=0)
    aspect: tuple[float, float] = (9, 16)


class WanI2VInput(AppInput):
    negative_prompt: str = ""
    megapixels: float = Field(default=0.2, gt=0)
    segment_frames: int | None = Field(default=None, ge=9)
    base_detection_prompt: str = "girl,"
    extension_detection_prompt: str = "girl, handkerchief"
    reference_detection_prompt: str = "human"


class AppLoadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    turbo: bool | None = None
    steps: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class UploadRule:
    media_type: str
    required: bool = False
    multiple: bool = False


@dataclass(frozen=True)
class AppDefinition:
    name: str
    module: str
    version: str
    input_model: type[AppInput]
    uploads: dict[str, UploadRule]
    uses_cpu_offload: bool = True
    gpu_modes: tuple[str, ...] = ("single_cuda",)

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        try:
            return AppLoadConfig.model_validate(config).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise ValueError("Invalid app load configuration") from exc

    def build_payload(self, payload: dict[str, Any], uploads: dict[str, list[Path]], output_path: Path) -> dict[str, Any]:
        try:
            result = self.input_model.model_validate(payload).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise ValueError("Invalid inference payload") from exc
        for name, rule in self.uploads.items():
            files = uploads.get(name, [])
            if rule.required and not files:
                raise ValueError(f"Missing required upload: {name}")
            if not rule.multiple and len(files) > 1:
                raise ValueError(f"Only one upload is allowed for {name}")
        if self.name == "mmh3_fi2v":
            result["first_frame"] = str(uploads["first_frame"][0])
            if uploads.get("last_frame"):
                result["last_frame"] = str(uploads["last_frame"][0])
        elif self.name == "mmh3_fr2v":
            images = uploads.get("reference_images", [])
            video = uploads.get("reference_video", [])
            audio = uploads.get("reference_audio", [])
            if not images and not video and not audio:
                raise ValueError("Provide at least one reference upload")
            result["reference_images"] = [str(path) for path in images]
            if video:
                result["reference_video"] = str(video[0])
            if audio:
                result["reference_audio"] = str(audio[0])
        else:
            result["reference_image"] = str(uploads["reference_image"][0])
            result["driving_video"] = str(uploads["driving_video"][0])
        result["output_path"] = str(output_path)
        return result


APPS = {
    "mmh3_fi2v": AppDefinition(
        name="mmh3_fi2v",
        module="apps.mmh3_fi2v",
        version="1",
        input_model=MiniMaxI2VInput,
        uploads={
            "first_frame": UploadRule("image", required=True),
            "last_frame": UploadRule("image"),
        },
    ),
    "mmh3_fr2v": AppDefinition(
        name="mmh3_fr2v",
        module="apps.mmh3_fr2v",
        version="1",
        input_model=MiniMaxR2VInput,
        uploads={
            "reference_images": UploadRule("image", multiple=True),
            "reference_video": UploadRule("video"),
            "reference_audio": UploadRule("audio"),
        },
    ),
    "wan_p_iv2v": AppDefinition(
        name="wan_p_iv2v",
        module="apps.wan_p_iv2v",
        version="1",
        input_model=WanI2VInput,
        uploads={
            "reference_image": UploadRule("image", required=True),
            "driving_video": UploadRule("video", required=True),
        },
    ),
}


def get_app(name: str) -> AppDefinition:
    try:
        return APPS[name]
    except KeyError:
        raise ValueError("Unsupported Flexy app") from None
