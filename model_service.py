import asyncio
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ─── Disease Database ─────────────────────────────────────────────────────────
DISEASE_DATABASE = {
    "healthy": {
        "name": "Feuille Saine",
        "severity": "none",
        "symptoms": "No disease detected. The leaf shows normal green coloration with no spots, discoloration, or deformations.",
        "cause": "The olive tree is in good health with no pathogen or nutrient deficiency detected.",
        "treatment": [
            "Continue regular agricultural practices",
            "Maintain balanced irrigation schedule",
            "Apply preventive copper-based spray in autumn",
            "Monitor regularly for early signs of disease",
        ],
        "prevention": [
            "Prune annually to improve air circulation",
            "Avoid over-irrigation",
            "Apply balanced NPK fertilization",
            "Monitor for insect pests regularly",
        ],
    },
    "peacock_spot": {
        "name": "Œil de Paon (Cycloconium)",
        "severity": "moderate",
        "symptoms": "Circular brownish spots with a yellow halo on the upper surface of leaves. Spots range from 2-10mm diameter.",
        "cause": "Caused by the fungus Spilocaea oleagina. Develops in humid, mild weather (10-20°C). Spores spread via rain splashes.",
        "treatment": [
            "Spray copper-based fungicide (Bordeaux mixture 1%) immediately",
            "Apply in autumn before rains and again in spring",
            "Remove and destroy fallen infected leaves",
            "Avoid overhead irrigation",
            "Apply systemic fungicides (Tebuconazole) for severe infections",
        ],
        "prevention": [
            "Treat with copper hydroxide every autumn",
            "Improve pruning to reduce humidity inside canopy",
            "Use certified disease-free plant material",
            "Avoid planting in poorly-drained soils",
        ],
    },
    "aculus_olearius": {
        "name": "Aculus Olearius (Acarien)",
        "severity": "moderate",
        "symptoms": "Silver-grey discoloration on leaf surface, leaf curling and deformation, stunted new growth, bronzing of affected tissue.",
        "cause": "Microscopic mite Aculus olearius that feeds on leaf surface cells. Populations explode in hot, dry conditions.",
        "treatment": [
            "Apply sulfur-based acaricide (wettable sulfur 0.3%) in early spring",
            "Use Abamectin or Bifenazate for severe infestations",
            "Apply when mite populations first detected (before bud break)",
            "Repeat treatment after 15-20 days if necessary",
            "Avoid broad-spectrum insecticides that kill natural predators",
        ],
        "prevention": [
            "Introduce predatory mites (Typhlodromus phialatus)",
            "Avoid excess nitrogen fertilization",
            "Apply preventive sulfur in late winter",
            "Monitor leaves with magnifying glass from March onwards",
        ],
    },
    "fumagina": {
        "name": "Fumagine (Suie / Noir des feuilles)",
        "severity": "moderate",
        "symptoms": "Black powdery coating on leaf surface (like soot), sticky residue, reduced photosynthesis, leaves appear dirty/blackened.",
        "cause": "Saprophytic fungi (Capnodium oleophilum) growing on honeydew secreted by scale insects (Saissetia oleae).",
        "treatment": [
            "First: control scale insects with white mineral oil (1-2%) in winter",
            "Apply insecticide (Chlorpyrifos or Dimethoate) against scale insects",
            "Wash leaves with water + soap solution to remove fungal coating",
            "Apply copper-based fungicide after insect control",
            "Prune heavily infested branches",
        ],
        "prevention": [
            "Monitor for scale insects from April onwards",
            "Apply white oil preventively in winter",
            "Maintain good canopy ventilation through pruning",
            "Avoid excess nitrogen that promotes soft growth attractive to insects",
        ],
    },
    "virosis": {
        "name": "Virosis (Maladie Virale)",
        "severity": "severe",
        "symptoms": "Mosaic pattern on leaves (yellow-green patches), leaf distortion, stunted growth, ring spots, vein yellowing.",
        "cause": "Various olive viruses (Olive latent virus 1 & 2, Arabis mosaic virus). Transmitted by nematodes, grafting tools.",
        "treatment": [
            "No direct cure for viral diseases",
            "Remove and destroy severely infected trees",
            "Disinfect all pruning tools with 70% alcohol or bleach solution",
            "Control nematode vectors with soil fumigation",
            "Apply balanced fertilization to support tree immunity",
        ],
        "prevention": [
            "Use only certified virus-free plant material",
            "Disinfect all cutting and grafting tools",
            "Control nematode populations in soil",
            "Quarantine new plant material before introduction",
            "Avoid grafting from infected trees",
        ],
    },
    "nutritional_deficiencies": {
        "name": "Carence Nutritionnelle",
        "severity": "mild",
        "symptoms": "Yellowing between leaf veins (interveinal chlorosis), pale green leaves, leaf tip burn, small leaves, reduced growth.",
        "cause": "Deficiency of Iron (Fe), Nitrogen (N), Magnesium (Mg), Boron (B), or Zinc (Zn). Often caused by incorrect soil pH.",
        "treatment": [
            "Analyze soil to identify specific deficiency",
            "For iron deficiency: apply iron chelate (EDDHA) to soil",
            "For nitrogen deficiency: apply urea or ammonium nitrate",
            "For magnesium: apply magnesium sulfate foliar spray",
            "For boron: apply borax 0.2% foliar spray in spring",
            "Correct soil pH to 6.0-7.5 for optimal nutrient absorption",
        ],
        "prevention": [
            "Perform annual soil and leaf analysis",
            "Apply balanced NPK + micronutrient fertilization",
            "Maintain proper irrigation to avoid nutrient leaching",
            "Use organic matter to improve soil structure",
        ],
    },
}

# ─── Class labels — MUST match exactly what train.py used ────────────────────
# These are the labels sorted alphabetically (how prepare_dataset.py created them)
CLASS_LABELS = [
    "aculus_olearius",
    "fumagina",
    "healthy",
    "nutritional_deficiencies",
    "peacock_spot",
    "virosis",
]


class OliveModelService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.is_loaded = False
        self._use_fallback = False

    async def load_model(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self):
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            import torch
            import os

            model_path = "./olive_model"
            logger.info(f"Loading fine-tuned model from: {model_path}")

            self.processor = ViTImageProcessor.from_pretrained(model_path)
            self.model = ViTForImageClassification.from_pretrained(
                model_path,
                num_labels=len(CLASS_LABELS),
                id2label={i: label for i, label in enumerate(CLASS_LABELS)},
                label2id={label: i for i, label in enumerate(CLASS_LABELS)},
                ignore_mismatched_sizes=True,
            )
            self.model.eval()
            self._use_fallback = False
            self.is_loaded = True
            logger.info(f"✅ Loaded specialized olive model: {model_path}")
            logger.info(f"Classes: {CLASS_LABELS}")

        except Exception as e:
            logger.error(f"Model loading error: {e}. Using color fallback.")
            self._use_fallback = True
            self.is_loaded = True

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, image)

    def _predict_sync(self, image: Image.Image) -> Dict[str, Any]:
        if self._use_fallback:
            return self._fallback_prediction(image)

        try:
            import torch

            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]

            probs_np = probs.numpy()
            pred_idx = int(np.argmax(probs_np))
            confidence = float(probs_np[pred_idx]) * 100

            # Direct mapping — index → CLASS_LABELS
            pred_key = CLASS_LABELS[pred_idx]

            # Build probabilities dict
            all_probs = {}
            for i, label in enumerate(CLASS_LABELS):
                all_probs[label] = round(float(probs_np[i]) * 100, 2)

            logger.info(f"Prediction: {pred_key} ({confidence:.1f}%)")
            logger.info(f"All probs: {all_probs}")

            return self._build_result(pred_key, confidence, all_probs)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._fallback_prediction(image)

    def _fallback_prediction(self, image: Image.Image) -> Dict[str, Any]:
        """Color-based fallback when model unavailable."""
        img_array = np.array(image.convert("RGB"))
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        avg_r, avg_g, avg_b = float(r.mean()), float(g.mean()), float(b.mean())
        dark_ratio = float((img_array.mean(axis=2) < 80).mean())
        yellow_ratio = float(((r > 150) & (g > 130) & (b < 100)).mean())
        silver_ratio = float(((r > 160) & (g > 160) & (b > 160) &
                               (np.abs(r.astype(int) - g.astype(int)) < 20)).mean())

        if dark_ratio > 0.15:
            pred_key, confidence = "fumagina", min(70 + dark_ratio * 100, 88)
        elif yellow_ratio > 0.2:
            pred_key, confidence = "nutritional_deficiencies", min(65 + yellow_ratio * 80, 85)
        elif silver_ratio > 0.3:
            pred_key, confidence = "aculus_olearius", min(68 + silver_ratio * 70, 87)
        elif avg_g > avg_r and avg_g > avg_b and avg_g > 80:
            pred_key, confidence = "healthy", 72.0
        else:
            pred_key, confidence = "peacock_spot", 62.0

        all_probs = {k: round((100 - confidence) / (len(CLASS_LABELS) - 1), 2) for k in CLASS_LABELS}
        all_probs[pred_key] = round(confidence, 2)

        return self._build_result(pred_key, confidence, all_probs)

    def _build_result(self, disease_key: str, confidence: float, all_probs: Dict) -> Dict[str, Any]:
        if disease_key not in DISEASE_DATABASE:
            disease_key = "healthy"

        info = DISEASE_DATABASE[disease_key]
        severity = info["severity"]
        if confidence < 55:
            severity = "uncertain"

        return {
            "disease_key": disease_key,
            "disease_name": info["name"],
            "confidence": round(confidence, 1),
            "severity": severity,
            "symptoms": info["symptoms"],
            "cause": info["cause"],
            "treatment": info["treatment"],
            "prevention": info["prevention"],
            "all_probabilities": all_probs,
        }
