import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import financial_news as n
from scripts import financial_baselines as b
from test_financial_baselines import sample as market_sample


def policy():
    p = json.loads((b.ROOT / n.POLICY).read_text())
    p.update(expected_training_rows=3, expected_validation_rows=2,
             minimum_training_rows=2, alpha_grid=[0.1, 10.0])
    return p


def sample():
    train, validation = market_sample()
    for i, r in enumerate(train + validation):
        r.update(candidate_count=4, candidate_empty=False)
        r['features'].update(news_volume=float(np.log1p(i % 3)),
                             threat_share=(i % 3) / 4, disruption_share=(i % 2) / 4)
    return train, validation


def baseline_predictions(train, validation, p):
    forecasts, _ = b.walk_forward(train, validation, p)
    return b.score_validation(forecasts, validation, p)[2]


class FinancialNewsTests(unittest.TestCase):
    def test_expanding_maturity_and_model_specific_training_scaling(self):
        train, validation = sample()
        validation[0]['features']['news_volume'] = float(np.log(5))
        forecasts, audit = n.walk_forward(train, validation, policy())
        self.assertEqual([len(r['training_targets']) for r in audit], [3, 4])
        self.assertEqual(len(forecasts), 8)
        for origin in audit:
            self.assertLess(b.utc(origin['latest_training_maturity']), b.utc(origin['cutoff_exclusive_utc']))
        for fit in audit[0]['fits']:
            fields = n.MODEL_FEATURES[fit['model']]
            self.assertEqual(fit['features'], fields)
            x = np.array([[r['features'][f] for f in fields] for r in train])
            np.testing.assert_allclose(fit['scaler_mean'], x.mean(axis=0))
            np.testing.assert_allclose(fit['scaler_variance'], x.var(axis=0))

    def test_current_and_future_outcomes_do_not_change_earlier_fits(self):
        train, validation = sample()
        original = n.walk_forward(train, validation, policy())
        changed = copy.deepcopy(validation)
        changed[-1]['target_ovx_mean'] = 99999.
        self.assertEqual(original, n.walk_forward(train, changed, policy()))
        changed[0]['target_ovx_mean'] = 88888.
        forecasts, audit = n.walk_forward(train, changed, policy())
        self.assertEqual(original[1][0], audit[0])
        self.assertEqual(original[0][:4], forecasts[:4])
        self.assertNotEqual(original[0][4:], forecasts[4:])

    def test_event_features_only_enter_M3(self):
        train, validation = sample()
        original, _ = n.walk_forward(train, validation, policy())
        for r in train + validation:
            r['features'].update(threat_share=1., disruption_share=1.)
        changed, _ = n.walk_forward(train, validation, policy())
        self.assertEqual([r for r in original if r['model'] == 'M2'],
                         [r for r in changed if r['model'] == 'M2'])
        self.assertNotEqual([r for r in original if r['model'] == 'M3'],
                            [r for r in changed if r['model'] == 'M3'])

    def test_empty_weeks_constant_features_and_overlapping_shares(self):
        train, validation = sample()
        for r in train + validation:
            r.update(candidate_count=0, candidate_empty=True)
            r['features'].update(news_volume=0., threat_share=0., disruption_share=0.)
        _, audit = n.walk_forward(train, validation, policy())
        for fit in audit[0]['fits']:
            self.assertTrue(all(s == 1 for s in fit['scaler_scale'][2:]))
        self.assertEqual(n.training_diagnostics(train)['constant_features'],
                         ['news_volume', 'threat_share', 'disruption_share'])
        train, validation = sample()
        train[0]['features'].update(threat_share=1., disruption_share=1.)
        n.validate_rows(train, validation, policy())

    def test_invalid_news_and_temporal_rows_rejected(self):
        for field, value in [('news_volume', -1), ('news_volume', 100), ('news_volume', float('nan')),
                             ('threat_share', 1.1), ('disruption_share', -0.1)]:
            train, validation = sample()
            validation[0]['features'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                n.validate_rows(train, validation, policy())
        for field, value in [('candidate_count', -1), ('candidate_count', True),
                             ('candidate_empty', True), ('split', 'test')]:
            train, validation = sample()
            validation[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                n.validate_rows(train, validation, policy())
        train, validation = sample()
        validation[0].update(candidate_count=0, candidate_empty=True)
        validation[0]['features'].update(news_volume=0, threat_share=.25)
        with self.assertRaisesRegex(ValueError, 'empty-week'):
            n.validate_rows(train, validation, policy())

    def test_json_models_match_independent_normal_equations(self):
        train, validation = sample()
        for model, fields in n.MODEL_FEATURES.items():
            for alpha in policy()['alpha_grid']:
                fit = json.loads(json.dumps(n.fit_model(train, model, alpha, policy())))
                x = np.array([[r['features'][f] for f in fields] for r in train])
                y = np.array([r['target_ovx_mean'] for r in train])
                scale = x.std(axis=0); scale[scale == 0] = 1
                z = (x - x.mean(axis=0)) / scale
                coef = np.linalg.solve(z.T @ z + alpha * np.eye(len(fields)), z.T @ (y - y.mean()))
                for row in validation:
                    expected = ((np.array([row['features'][f] for f in fields]) - x.mean(axis=0)) / scale) @ coef + y.mean()
                    self.assertAlmostEqual(b.predict_saved(fit, row), expected, places=10)

    def test_independent_alpha_selection_paired_metrics_and_frozen_baselines(self):
        train, validation = sample(); p = policy()
        forecasts, _ = n.walk_forward(train, validation, p)
        for r in forecasts:
            actual = next(v['target_ovx_mean'] for v in validation if v['target_week_start'] == r['target_week_start'])
            # M2 prefers .1; exact M3 ties prefer 10.
            r['prediction'] = actual + (2 if r['model'] == 'M2' and r['alpha'] == 10 else 1)
        baselines = baseline_predictions(train, validation, p)
        summary, _, chosen = n.score_validation(forecasts, baselines, validation, p)
        self.assertEqual(summary['selected_alphas'], {'M2': .1, 'M3': 10.})
        self.assertEqual(summary['comparisons']['M3_minus_M2']['mae_difference'], 0)
        self.assertEqual([r for r in chosen if r['model'] in ('M0','M1')], baselines)
        self.assertFalse(summary['final_test_scored'])
        for result in summary['models'].values(): self.assertEqual(result['weeks'], 2)
        self.assertIsNone(summary['quarters']['2']['M3']['mae'])

    def test_missing_duplicate_misaligned_or_changed_actuals_rejected(self):
        train, validation = sample(); p = policy()
        forecasts, _ = n.walk_forward(train, validation, p)
        baselines = baseline_predictions(train, validation, p)
        for bad in [forecasts[:-1], forecasts + forecasts[:1]]:
            with self.assertRaisesRegex(ValueError, 'forecasts'):
                n.score_validation(bad, baselines, validation, p)
        for bad in [baselines[:-1], baselines + baselines[:1]]:
            with self.assertRaisesRegex(ValueError, 'forecasts'):
                n.score_validation(forecasts, bad, validation, p)
        for field, value in [('actual', 999.), ('cutoff_exclusive_utc', '2023-01-03T00:00:00+00:00')]:
            bad = copy.deepcopy(baselines); bad[0][field] = value
            with self.assertRaises(ValueError): n.score_validation(forecasts, bad, validation, p)

    def test_real_loader_parses_only_allowed_inputs_and_rejects_protocol_drift(self):
        p = policy()
        train, validation = sample()
        baselines = baseline_predictions(train, validation, p)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config').mkdir()
            (root / 'scripts').mkdir()
            # Preserve real provenance checks using source code and synthetic data.
            for name in ('financial_news.py', 'financial_baselines.py',
                         'prepare_news_review.py', 'prepare_modelling_dataset.py',
                         'reserve_news_evaluation.py', 'validate_evaluation_annotations.py'):
                (root / 'scripts' / name).write_bytes((b.ROOT / 'scripts' / name).read_bytes())
            dataset = root / p['dataset_directory']
            dataset.mkdir(parents=True)
            b.write_rows(dataset / 'primary_training.jsonl', train)
            b.write_rows(dataset / 'primary_validation.jsonl', validation)
            # Hashing is allowed; parsing this held-out sentinel must fail.
            (dataset / 'primary_test.jsonl').write_text('DO NOT PARSE HELD-OUT DATA\n')
            b.save_json(dataset / 'manifest.json', {
                'input_sha256': {},
                'artifact_sha256': {name: b.digest(dataset / name) for name in (
                    'primary_training.jsonl', 'primary_validation.jsonl', 'primary_test.jsonl')},
            })
            p['dataset_manifest_sha256'] = b.digest(dataset / 'manifest.json')
            b.save_json(root / b.POLICY, p)
            baseline_dir = root / p['baseline_directory']
            baseline_dir.mkdir(parents=True)
            b.write_rows(baseline_dir / 'selected_predictions.jsonl', baselines)
            b.save_json(baseline_dir / 'manifest.json', {
                'input_sha256': {b.POLICY: b.digest(root / b.POLICY)},
                'artifact_sha256': {
                    'selected_predictions.jsonl': b.digest(baseline_dir / 'selected_predictions.jsonl')},
            })
            p['baseline_manifest_sha256'] = b.digest(baseline_dir / 'manifest.json')
            b.save_json(root / n.POLICY, p)
            with patch.object(b, 'read', wraps=b.read) as reads:
                loaded = n.load_inputs(root, p)
            self.assertEqual([c.args[0].name for c in reads.call_args_list],
                             ['primary_training.jsonl', 'primary_validation.jsonl', 'selected_predictions.jsonl'])
            self.assertEqual(loaded[:3], (train, validation, baselines))
            changed = copy.deepcopy(p)
            changed['alpha_grid'] = [1000.]
            with self.assertRaisesRegex(ValueError, 'baseline design'):
                n.load_inputs(root, changed)
            (dataset / 'primary_test.jsonl').write_text('TAMPERED\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                n.load_inputs(root, p)

    def test_final_snapshots_and_paired_errors(self):
        train, validation = sample(); p = policy()
        baselines = baseline_predictions(train, validation, p)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            with patch.object(n, 'load_inputs', return_value=(train,validation,baselines,{})):
                summary, _ = n.compute(stage,stage,p)
            final = json.loads((stage/'final_models.json').read_text())
            for model, fit in final['models'].items():
                self.assertEqual(fit['training_rows'], 5)
                self.assertEqual(fit['training_last_target_end'], '2023-01-13')
                self.assertEqual(fit['alpha'], summary['selected_alphas'][model])
            paired = b.read(stage/'paired_errors.jsonl')
            self.assertAlmostEqual(np.mean([r['absolute_error_M3_minus_M2'] for r in paired]),
                                   summary['comparisons']['M3_minus_M2']['mae_difference'])

    def test_snapshot_refuses_overwrite_reproduces_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'config').mkdir(); p = policy()
            (root/n.POLICY).write_text(json.dumps(p))
            def compute(stage, root, policy):
                b.save_json(stage/'summary.json', {'complete': True})
                return {'complete':True}, {n.POLICY:b.digest(root/n.POLICY)}
            with patch.object(n, 'compute', side_effect=compute):
                self.assertEqual(n.run(root), n.run(root, True))
                with self.assertRaisesRegex(ValueError, 'exists'): n.run(root)
                (root/p['output_directory']/'summary.json').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'): n.run(root,True)


if __name__ == '__main__':
    unittest.main()
