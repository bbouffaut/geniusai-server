# Style-Match develop-settings whitelist

Canonical classification of PV2012 develop-settings keys for the style
transfer (plan §2.9). The plugin's `StyleWhitelist.lua` mirrors this file; the
empirical T0.4 diff (move every slider, diff `getDevelopSettings()` before and
after) refines both — keep them in sync.

| Layer | Keys | Transfer |
|---|---|---|
| **Style** | `Contrast2012`, `Highlights2012`, `Shadows2012`, `Clarity2012`, `Texture`, `Dehaze`, `Vibrance`, `Saturation`, `Parametric*`, `ToneCurvePV2012*`, `SplitToning*`, `ColorGrade*`, `Hue*`/`Saturation*`/`Luminance*` (HSL), `GrayMix*`, `ConvertToGrayscale`, `Sharpen*`, `NoiseReduction*`, `ColorNoiseReduction*`, `PostCropVignette*`, `Grain*`, `CameraProfile`, `LookName`, `LookAmount`, `ShadowTint`, `HighlightTint` | ✅ default ("Apply style") |
| **Scene compensation** | `Exposure2012`, `Whites2012`, `Blacks2012`, `WhiteBalance`, `Temperature`, `Tint` | ⚠️ opt-in ("Apply style + exposure/WB") |
| **Local / geometry** | `LocalAdjustments`, `RetouchInfo`, `AutoLateralCA`, `Orientation`, `Crop*`, `Perspective*`, `Correction*` | ❌ never — stripped server-side, count reported as `local_adjustments` |

Known gaps (filled by T0.4): `Texture` and the local-correction key names are
undocumented in the SDK; exact colour-grading key names differ between Lr
versions. `StyleMatchSpike.lua` + `styleMatchDiffDevelopSettings()` produce the
empirical diff; update both whitelist files from its output.
