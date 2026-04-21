import asyncio
import logging
import numpy as np
from PIL import Image
import io
import os
import base64
import httpx
from typing import Tuple, Optional, List
from schemas import BoundingBox

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)

# ── Roboflow config ──────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "0VNpMt8MtgSej0zsuPGf")

ROBOFLOW_MODEL_1   = "olive-leaf-7c6hz"
ROBOFLOW_VERSION_1 = 2

ROBOFLOW_MODEL_2   = "olive-leaf-detection"
ROBOFLOW_VERSION_2 = 1

CONFIDENCE_THRESHOLD = 0.40


class ImageProcessor:
    """
    4-layer leaf detection pipeline:
      1. Roboflow model 1  (2000-image YOLO — primary)
      2. Roboflow model 2  (107-image YOLO — fallback)
      3. OpenCV HSV        (color-based fallback)
      4. Center crop       (last resort)

    After detection → white background normalization
    (matches training data which was all white/grey background)
    """

    async def detect_and_crop_leaf(
        self, image_bytes: bytes
    ) -> Tuple[Image.Image, bool, Optional[BoundingBox]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_sync, image_bytes)

    def _process_sync(
        self, image_bytes: bytes
    ) -> Tuple[Image.Image, bool, Optional[BoundingBox]]:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = pil_image.size

            # ── Layer 1 : Roboflow model 1 ──────────────────────────────
            detections = self._call_roboflow(
                image_bytes, ROBOFLOW_MODEL_1, ROBOFLOW_VERSION_1
            )

            # ── Layer 2 : Roboflow model 2 ──────────────────────────────
            if not detections:
                logger.info("Model 1 found nothing → trying model 2")
                detections = self._call_roboflow(
                    image_bytes, ROBOFLOW_MODEL_2, ROBOFLOW_VERSION_2
                )

            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                logger.info(f"✅ Roboflow leaf detected conf={best['confidence']:.2f}")
                cropped, bbox = self._crop_detection(pil_image, best, w, h)
                # ✅ Normalize background to white (match training data)
                normalized = self._normalize_background(cropped)
                return self._preprocess(normalized), True, bbox

            # ── Layer 3 : OpenCV fallback ───────────────────────────────
            logger.warning("Roboflow found nothing → OpenCV fallback")
            cropped, bbox = self._opencv_fallback(pil_image, w, h)
            if cropped is not None:
                normalized = self._normalize_background(cropped)
                return self._preprocess(normalized), False, bbox

            # ── Layer 4 : center crop ───────────────────────────────────
            logger.warning("OpenCV failed → center crop")
            cropped, bbox = self._center_crop(pil_image, w, h)
            normalized = self._normalize_background(cropped)
            return self._preprocess(normalized), False, bbox

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return self._preprocess(img), False, None

    # ── White Background Normalization ───────────────────────────────────────

    def _normalize_background(self, image: Image.Image) -> Image.Image:
        """
        Isolate the leaf and place it on a white background.
        This matches the training dataset (white/grey background).
        
        Steps:
          1. Convert to HSV
          2. Detect leaf pixels (green/yellow/brown range)
          3. Create mask
          4. Place leaf on white background
        """
        try:
            import cv2

            img_array = np.array(image.convert("RGB"))
            h, w = img_array.shape[:2]

            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Detect leaf colors: green, yellow-green, brown, dark
            masks = [
                # Green leaves (healthy, aculus)
                cv2.inRange(hsv, np.array([25, 15, 15]), np.array([100, 255, 255])),
                # Yellow-brown (peacock spot, nutritional)
                cv2.inRange(hsv, np.array([10, 20, 30]), np.array([35, 255, 200])),
                # Dark brown/black (fumagina, virosis)
                cv2.inRange(hsv, np.array([0,  0,  10]), np.array([180, 80,  80])),
                # Grey-silver (aculus olearius)
                cv2.inRange(hsv, np.array([0,  0,  80]), np.array([180, 30, 200])),
            ]

            # Combine all masks
            mask = masks[0]
            for m in masks[1:]:
                mask = cv2.bitwise_or(mask, m)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

            # Keep only largest contour (the leaf)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                clean_mask = np.zeros_like(mask)
                largest = max(contours, key=cv2.contourArea)
                # Only use mask if leaf is significant part of image
                if cv2.contourArea(largest) > w * h * 0.05:
                    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
                    mask = clean_mask
                else:
                    # Leaf too small — use full image
                    mask = np.ones((h, w), dtype=np.uint8) * 255

            # Create white background
            white_bg = np.ones_like(img_array) * 255

            # Blend: leaf pixels from original, rest white
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
            result = (img_array * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)

            return Image.fromarray(result)

        except Exception as e:
            logger.warning(f"Background normalization failed: {e} — using original")
            return image

    # ── Roboflow API ─────────────────────────────────────────────────────────

    def _call_roboflow(
        self, image_bytes: bytes, model_id: str, version: int
    ) -> List[dict]:
        try:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            url = (
                f"https://detect.roboflow.com/{model_id}/{version}"
                f"?api_key={ROBOFLOW_API_KEY}"
                f"&confidence={int(CONFIDENCE_THRESHOLD * 100)}"
                f"&overlap=30"
            )
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    url,
                    content=img_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            if resp.status_code != 200:
                logger.warning(f"Roboflow {model_id} → HTTP {resp.status_code}")
                return []
            preds = resp.json().get("predictions", [])
            return [p for p in preds if p.get("confidence", 0) >= CONFIDENCE_THRESHOLD]
        except httpx.TimeoutException:
            logger.warning(f"Roboflow {model_id} timeout")
            return []
        except Exception as e:
            logger.warning(f"Roboflow {model_id} error: {e}")
            return []

    def _crop_detection(
        self, image: Image.Image, det: dict, w: int, h: int
    ) -> Tuple[Image.Image, BoundingBox]:
        cx, cy, bw, bh = det["x"], det["y"], det["width"], det["height"]
        pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
        x1 = max(0, int(cx - bw / 2) - pad_x)
        y1 = max(0, int(cy - bh / 2) - pad_y)
        x2 = min(w, int(cx + bw / 2) + pad_x)
        y2 = min(h, int(cy + bh / 2) + pad_y)
        bbox = BoundingBox(x=x1/w, y=y1/h, width=(x2-x1)/w, height=(y2-y1)/h)
        return image.crop((x1, y1, x2, y2)), bbox

    # ── OpenCV fallback ──────────────────────────────────────────────────────

    def _opencv_fallback(
        self, pil_image: Image.Image, w: int, h: int
    ) -> Tuple[Optional[Image.Image], Optional[BoundingBox]]:
        try:
            import cv2
            img = np.array(pil_image)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            masks = [
                cv2.inRange(hsv, np.array([20, 20, 20]),  np.array([100, 255, 255])),
                cv2.inRange(hsv, np.array([15, 30, 50]),  np.array([35,  255, 255])),
                cv2.inRange(hsv, np.array([5,  30, 30]),  np.array([25,  255, 180])),
            ]
            mask = masks[0]
            for m in masks[1:]:
                mask = cv2.bitwise_or(mask, m)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None, None
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < w * h * 0.03:
                return None, None
            x, y, cw, ch = cv2.boundingRect(largest)
            px, py = int(cw * 0.10), int(ch * 0.10)
            x1 = max(0, x-px);  y1 = max(0, y-py)
            x2 = min(w, x+cw+px); y2 = min(h, y+ch+py)
            bbox = BoundingBox(x=x1/w, y=y1/h, width=(x2-x1)/w, height=(y2-y1)/h)
            return Image.fromarray(img[y1:y2, x1:x2]), bbox
        except Exception as e:
            logger.warning(f"OpenCV error: {e}")
            return None, None

    # ── Center crop ──────────────────────────────────────────────────────────

    def _center_crop(
        self, image: Image.Image, w: int, h: int
    ) -> Tuple[Image.Image, BoundingBox]:
        size = min(w, h)
        left, top = (w - size) // 2, (h - size) // 2
        bbox = BoundingBox(x=left/w, y=top/h, width=size/w, height=size/h)
        return image.crop((left, top, left+size, top+size)), bbox

    # ── Preprocess ───────────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> Image.Image:
        return image.resize(TARGET_SIZE, Image.LANCZOS)
