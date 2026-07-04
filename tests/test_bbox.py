import pytest

from xdiff.core import main
from xdiff.model import BoundingBox


def test_bounding_box_accepts_valid_window():
    box = BoundingBox(lon_min=-6, lon_max=36, lat_min=30, lat_max=46)
    assert (box.lon_min, box.lon_max, box.lat_min, box.lat_max) == (-6, 36, 30, 46)


def test_bounding_box_rejects_non_increasing_longitude():
    with pytest.raises(ValueError, match="longitude min"):
        BoundingBox(lon_min=10, lon_max=5, lat_min=0, lat_max=1)


def test_bounding_box_rejects_non_increasing_latitude():
    with pytest.raises(ValueError, match="latitude min"):
        BoundingBox(lon_min=0, lon_max=1, lat_min=10, lat_max=5)


def test_bounding_box_rejects_latitude_out_of_range():
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        BoundingBox(lon_min=0, lon_max=1, lat_min=-100, lat_max=10)


def test_normalize_bbox_coerces_tuple_and_passes_through():
    box = main.normalize_bbox((-6, 36, 30, 46))
    assert isinstance(box, BoundingBox)
    assert main.normalize_bbox(box) is box
    assert main.normalize_bbox(None) is None
