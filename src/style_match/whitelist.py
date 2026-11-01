"""Develop-settings whitelist for the Style Match transfer (plan §2.9).

Canonical server-side copy of the classification in the plugin's
StyleWhitelist.lua (keep both in sync; the empirical T0.4 diff refines them).

Three layers of a develop-settings snapshot:
  style  – transferred by default ("Apply style")
  scene  – exposure / white balance compensations for THAT frame; opt-in
           ("Apply style + exposure/WB")
  local  – local corrections / geometry, tied to the reference composition;
           NEVER transferred and stripped from returned snapshots.
"""

STYLE_KEYS = {
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Clarity2012",
    "Texture",
    "Dehaze",
    "Vibrance",
    "Saturation",
    "ParametricShadows",
    "ParametricDarks",
    "ParametricLights",
    "ParametricHighlights",
    "ParametricShadowSplit",
    "ParametricMidtoneSplit",
    "ParametricLightSplit",
    "ParametricHighlightSplit",
    "ToneCurvePV2012",
    "ToneCurvePV2012Red",
    "ToneCurvePV2012Green",
    "ToneCurvePV2012Blue",
    "SplitToningHighlightHue",
    "SplitToningHighlightSaturation",
    "SplitToningBalance",
    "SplitToningShadowHue",
    "SplitToningShadowSaturation",
    "SharpenAmount",
    "SharpenRadius",
    "SharpenDetail",
    "SharpenEdgeMasking",
    "NoiseReductionAmount",
    "NoiseReductionDetail",
    "NoiseReductionColor",
    "ColorNoiseReduction",
    "ColorNoiseReductionDetail",
    "ColorNoiseReductionSmoothness",
    "PostCropVignetteAmount",
    "PostCropVignetteMidpoint",
    "PostCropVignetteFeather",
    "PostCropVignetteRoundness",
    "PostCropVignetteStyle",
    "PostCropVignetteContrast",
    "PostCropVignetteHighlightContrast",
    "GrainAmount",
    "GrainSize",
    "GrainFrequency",
    "CameraProfile",
    "LookName",
    "LookAmount",
    "ConvertToGrayscale",
    "ShadowTint",
    "HighlightTint",
}

# Prefix-matched style families: colour grading (ColorGrade*), HSL per-colour
# (Hue*/Saturation*/Luminance*), B&W mix (GrayMix*).
STYLE_KEY_PREFIXES = ("ColorGrade", "Hue", "Saturation", "Luminance", "GrayMix")

SCENE_KEYS = {
    "Exposure2012",
    "Whites2012",
    "Blacks2012",
    "WhiteBalance",
    "Temperature",
    "Tint",
}

# Local corrections + geometry: never transferred, stripped from snapshots.
LOCAL_KEYS = {
    "LocalAdjustments",
    "RetouchInfo",
    "AutoLateralCA",
    "Orientation",
}
LOCAL_KEY_PREFIXES = ("Crop", "Perspective", "Correction")

SIGNAL_KEYS = ("sem", "desc", "color", "loc")


def is_style_key(key):
    return key in STYLE_KEYS or str(key).startswith(STYLE_KEY_PREFIXES)


def is_local_key(key):
    return key in LOCAL_KEYS or str(key).startswith(LOCAL_KEY_PREFIXES)


def style_subset(develop_settings):
    """Return only the style keys present in the snapshot (Apply default)."""
    if not isinstance(develop_settings, dict):
        return {}
    return {key: value for key, value in develop_settings.items() if is_style_key(key)}


def scene_subset(develop_settings):
    """Return the exposure/WB compensation keys (opt-in toggle)."""
    if not isinstance(develop_settings, dict):
        return {}
    return {key: value for key, value in develop_settings.items() if key in SCENE_KEYS}


def strip_local_adjustments(develop_settings):
    """Strip local/geometry keys; return (clean_snapshot, local_count)."""
    if not isinstance(develop_settings, dict):
        return {}, 0
    clean = {}
    local_count = 0
    for key, value in develop_settings.items():
        if is_local_key(key):
            local_count += 1
            continue
        clean[key] = value
    return clean, local_count
