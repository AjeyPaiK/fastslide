# Copyright 2026 Ajey Pai Karkala. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Human-readable labels from GeoJSON annotation files."""

from __future__ import annotations

from typing import Any


def geojson_title(geojson: dict[str, Any]) -> str | None:
    """Returns a display title for a GeoJSON annotation object.

    Checks, in order:

    - a top-level ``title`` string (when present), then
    - distinct ``classification.name`` values on features (QuPath export).
    """
    title = geojson.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    names: list[str] = []
    features = geojson.get("features")
    if not isinstance(features, list):
        return None
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        cls = props.get("classification")
        if not isinstance(cls, dict):
            continue
        name = cls.get("name")
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name.strip())
    return ", ".join(names) if names else None
