import tempfile
import time
import unittest
from unittest.mock import patch
from ripesense.demo import create_app, FALLBACK, validate_recipe, generate_recipe


class DemoTests(unittest.TestCase):
    def test_scan_qr_and_persistent_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory, lambda: (dict(FALLBACK), None), duration=.1)
            client = app.test_client()
            self.assertEqual(client.get('/api/demo/state').json['status'], 'idle')
            response = client.post('/api/demo/scan')
            self.assertEqual(response.status_code, 202)
            scan_id = response.json['scan_id']
            self.assertEqual(client.post('/api/demo/scan').status_code, 409)
            for _ in range(50):
                state = client.get('/api/demo/state').json
                if state['status'] == 'result' and state['recipe_status'] == 'ready':
                    break
                time.sleep(.02)
            self.assertEqual(state['scenario']['ideal_days'], '1–2')
            self.assertTrue(state['scenario']['simulated'])
            self.assertEqual(state['status'], 'result')
            self.assertEqual(client.get(f'/api/demo/qr/{scan_id}.svg').mimetype, 'image/svg+xml')
            self.assertEqual(client.get(f'/r/{scan_id}').status_code, 200)
            restarted = create_app(directory).test_client()
            self.assertEqual(restarted.get(f'/api/demo/recipe/{scan_id}').json['recipe']['title'], FALLBACK['title'])
            self.assertEqual(client.get('/api/demo/recipe/not-a-valid-id').status_code, 404)

    def test_remote_and_cross_origin_cannot_spend_key_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory).test_client()
            self.assertEqual(client.post('/api/demo/scan', environ_base={'REMOTE_ADDR': '10.1.2.3'}).status_code, 403)
            self.assertEqual(client.post('/api/demo/scan', headers={'Origin': 'https://other.example'}).status_code, 403)

    def test_recipe_validation_and_missing_key(self):
        validate_recipe(FALLBACK)
        with self.assertRaises(ValueError):
            validate_recipe({**FALLBACK, 'steps': 'not a list'})
        with patch('ripesense.demo.read_key', return_value=''):
            recipe, error = generate_recipe()
        self.assertEqual(recipe['source'], 'Built-in fallback')
        self.assertIn('not configured', error)

    def test_provider_failure_finishes_with_labeled_fallback(self):
        def failed():
            raise RuntimeError('offline')
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory, failed, duration=.01).test_client()
            scan_id = client.post('/api/demo/scan').json['scan_id']
            for _ in range(50):
                response = client.get(f'/api/demo/recipe/{scan_id}')
                if response.status_code == 200:
                    break
                time.sleep(.01)
            self.assertEqual(response.json['recipe']['source'], 'Built-in fallback')
            self.assertIsNotNone(response.json['recipe_error'])


if __name__ == '__main__':
    unittest.main()
