"""Pydantic models documenting the style-match API contract (plan §3.3).

The Flask routes parse multipart form values directly; these models pin the
JSON sub-payload shapes (weights / filters) and are used by tests.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class StyleMatchWeights(BaseModel):
    sem: float = 0.5
    desc: float = 0.3
    color: float = 0.1
    loc: float = 0.1


class StyleMatchFilters(BaseModel):
    # has_edits defaults to True: unedited references are hidden unless asked
    # for (plan §2.6). Date/location/camera are FILTERS, never scored.
    has_edits: Optional[bool] = True
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_km: Optional[float] = None
    camera_model: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class StyleMatchSearchRequest(BaseModel):
    top_k: int = 12
    weights: StyleMatchWeights = Field(default_factory=StyleMatchWeights)
    filters: StyleMatchFilters = Field(default_factory=StyleMatchFilters)
    scoring_mode: Literal["linear", "rrf"] = "linear"
    description: Optional[str] = None


class StyleMatchSignalScores(BaseModel):
    sem: Optional[float] = None
    desc: Optional[float] = None
    color: Optional[float] = None
    loc: Optional[float] = None


class StyleMatchMatch(BaseModel):
    rank: int
    lrc_local_id: int
    catalog_id: str
    uuid: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    score: float
    signals: StyleMatchSignalScores
    capture_time: Optional[str] = None
    distance_km: Optional[float] = None
    camera_model: Optional[str] = None
    has_edits: Optional[bool] = None
    description: Optional[str] = None
    develop_settings: Optional[dict] = None
    local_adjustments: int = 0


class StyleMatchSearchResponse(BaseModel):
    status: Literal["ok", "error"]
    query: dict = {}
    signals_used: list[str] = []
    scoring_mode: str = "linear"
    candidate_count: int = 0
    matches: list[StyleMatchMatch] = []
