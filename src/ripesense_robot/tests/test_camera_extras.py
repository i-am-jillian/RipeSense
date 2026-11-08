import csv

import pytest

from ripesense.camera_only import ROOT,decode_photo
from ripesense.models.camera_extras import ExtraFruitModels


@pytest.fixture(scope='module')
def models():
    return ExtraFruitModels(ROOT/'models/camera_extras')


@pytest.mark.parametrize('item',['avocado','tomato'])
def test_real_extra_head_matches_heldout_record(models,item):
    with (ROOT/f'models/camera_extras/{item}_test_predictions.csv').open() as f:
        row=next(csv.DictReader(f))
    result=models.predict(decode_photo((ROOT/row['path']).read_bytes()),item)
    assert result['fruit_type']==item
    assert result['predicted_stage']==row['predicted']
    assert sum(result['probabilities'].values())==pytest.approx(1)
    assert result['days_left'] is None
    assert 'Trained' in result['assessment_basis']
    if item=='avocado':assert 'Hass' in result['assessment_basis']
