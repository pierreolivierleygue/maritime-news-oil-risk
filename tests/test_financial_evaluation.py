import copy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_finance as e
from scripts import financial_baselines as b
from test_financial_baselines import row


def policy():
    p = json.loads((b.ROOT / e.POLICY).read_text())
    p['minimum_training_rows'] = 2
    p['panels'] = {k: v for k, v in p['panels'].items() if v['view'] == 'combined'}
    for panel in p['panels'].values():
        panel['expected_rows'] = {'training': 3, 'validation': 2, 'test': 8}
    p['bootstrap'].update(replicates=100, primary_block_weeks=2,
                           sensitivity_block_weeks=[1,4], development_validation_weeks=8)
    return p


def sample(conservative=False):
    rows = []
    days = [('2022-12-12','training'), ('2022-12-19','training'), ('2022-12-26','training'),
            ('2023-12-18','validation'), ('2023-12-25','validation')]
    days += [((date(2024,1,1) + i * timedelta(days=7)).isoformat(), 'test') for i in range(8)]
    for i, (day, split) in enumerate(days):
        r = row(day, split, [10. + i*2, 8. + i], 12. + i*2 + (i%3))
        r.update(candidate_count=4, candidate_empty=False, coverage=[{'accepted_observed_coverage':True}])
        r['features'].update(news_volume=float(np.log1p(i%3)), threat_share=(i%3)/4, disruption_share=(i%2)/4)
        if conservative:
            r['timing'] = 'conservative'
            r['ovx_history_weeks'] = [(date.fromisoformat(w) - timedelta(days=7)).isoformat() for w in r['ovx_history_weeks']]
        rows.append(r)
    return rows


class FinancialEvaluationTests(unittest.TestCase):
    def test_expanding_target_availability_and_fixed_penalties(self):
        p = policy()
        for name, panel in p['panels'].items():
            rows = sample(panel['timing'] == 'conservative')
            forecasts, audit = e.walk_forward(rows, panel, p)
            first = 4 if panel['timing'] == 'conservative' else 5
            self.assertEqual([len(r['training_targets']) for r in audit], list(range(first,first+8)))
            for origin in audit:
                self.assertLess(b.utc(origin['latest_training_available_at']), b.utc(origin['cutoff_exclusive_utc']))
                for fit in origin['fits']:
                    self.assertEqual(fit['alpha'], p['selected_alphas'][fit['model']])
            self.assertEqual(len(forecasts),32)
            self.assertEqual([r['prediction'] for r in forecasts if r['model']=='M0'],
                             [r['features']['ovx_latest'] for r in rows if r['split']=='test'])

    def test_strict_delayed_maturity_boundary(self):
        rows = sample()
        rows[0]['target_mature_after_utc'] = '2023-12-25T00:00:00+00:00'
        available = e.available_rows(rows,'2024-01-01T00:00:00+00:00',7)
        self.assertNotIn(rows[0],available)
        self.assertIn(rows[0],e.available_rows(rows,'2024-01-01T00:00:01+00:00',7))
        with self.assertRaises(ValueError): e.available_rows(rows,'2024-01-01T00:00:00+00:00',-1)

    def test_test_outcomes_cannot_affect_their_own_or_earlier_forecasts(self):
        p=policy()
        for name,panel in p['panels'].items():
            rows=sample(panel['timing']=='conservative')
            original=e.walk_forward(rows,panel,p)
            changed=copy.deepcopy(rows);changed[-1]['target_ovx_mean']=9999.
            self.assertEqual(original,e.walk_forward(changed,panel,p))
            changed[5]['target_ovx_mean']=8888.
            result=e.walk_forward(changed,panel,p)
            protected=2 if panel['timing']=='conservative' else 1
            self.assertEqual(original[0][:protected*4],result[0][:protected*4])
            self.assertNotEqual(original[0][protected*4:],result[0][protected*4:])

    def test_scaling_and_independent_normal_equations(self):
        p=policy();panel=p['panels']['publication/combined'];rows=sample()
        rows[5]['features']['ovx_latest']=1000.
        forecasts,audit=e.walk_forward(rows,panel,p)
        for fit in audit[0]['fits']:
            fields=fit['features'];x=np.array([[r['features'][f] for f in fields] for r in rows[:5]])
            y=np.array([r['target_ovx_mean'] for r in rows[:5]])
            np.testing.assert_allclose(fit['scaler_mean'],x.mean(axis=0))
            scale=x.std(axis=0);scale[scale==0]=1;z=(x-x.mean(axis=0))/scale
            coef=np.linalg.solve(z.T@z+fit['alpha']*np.eye(len(fields)),z.T@(y-y.mean()))
            test=np.array([rows[5]['features'][f] for f in fields])
            expected=((test-x.mean(axis=0))/scale)@coef+y.mean()
            actual=next(r['prediction'] for r in forecasts[:4] if r['model']==fit['model'])
            self.assertAlmostEqual(actual,expected,places=9)

    def test_source_news_changes_cannot_change_market_baselines(self):
        p=policy();panel=p['panels']['publication/combined'];rows=sample()
        original,_=e.walk_forward(rows,panel,p)
        for r in rows:r['features'].update(news_volume=0.,threat_share=0.,disruption_share=0.)
        changed,_=e.walk_forward(rows,panel,p)
        self.assertEqual([r for r in original if r['model'] in ('M0','M1')],
                         [r for r in changed if r['model'] in ('M0','M1')])

    def test_invalid_split_history_coverage_and_news_rejected(self):
        p=policy();panel=p['panels']['publication/combined']
        for field,value in [('split','validation'),('target_mature_after_utc','2024-01-01T00:00:00+00:00'),
                            ('target_observations',2),('target_ovx_mean',float('nan')),('candidate_count',-1),
                            ('coverage',[{'accepted_observed_coverage':False}])]:
            rows=sample();rows[5][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):e.validate_panel(rows,panel,p)
        for field,value in [('news_volume',100),('threat_share',-1),('disruption_share',float('nan'))]:
            rows=sample();rows[5]['features'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):e.validate_panel(rows,panel,p)
        with self.assertRaisesRegex(ValueError,'history'):e.validate_panel(sample(True),p['panels']['conservative/combined']|{'ovx_lag_weeks':0},p)
        with self.assertRaisesRegex(ValueError,'population'):e.validate_panel(sample()[:-1],panel,p)
        rows=sample();rows[6]=copy.deepcopy(rows[5])
        with self.assertRaisesRegex(ValueError,'Repeated'):e.validate_panel(rows,panel,p)

    def test_scoring_rejects_missing_duplicate_wrong_dates_and_retuning(self):
        p=policy();rows=sample();forecasts,_=e.walk_forward(rows,p['panels']['publication/combined'],p)
        for bad in [forecasts[:-1],forecasts+forecasts[:1]]:
            with self.assertRaisesRegex(ValueError,'forecasts'):e.score_forecasts(bad,rows,p)
        for field,value in [('alpha',10.),('target_week_end','2025-01-01'),('prediction',float('inf'))]:
            bad=copy.deepcopy(forecasts);bad[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):e.score_forecasts(bad,rows,p)
        scores=e.score_forecasts(forecasts,rows,p)
        metrics=e.summarize_scores(scores,p)
        paired=e.paired_errors(scores)
        self.assertAlmostEqual(np.mean([r['absolute_error_M3_minus_M2'] for r in paired]),
                               metrics['comparisons']['M3_minus_M2']['mae_difference'])
        self.assertAlmostEqual(paired[-1]['cumulative_absolute_error_M3_minus_M2'],
                               8*metrics['comparisons']['M3_minus_M2']['mae_difference'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):e.paired_errors(scores+scores)

    def test_panels_use_common_dates_and_matching_actuals(self):
        a=sample();other=sample(True)[:-1]
        common,omitted=e.common_test_weeks({'a':a,'b':other},'a')
        self.assertEqual(len(common),7)
        self.assertEqual(omitted['a'],[a[-1]['target_week_start']])
        self.assertEqual(omitted['b'],[])
        other[5]['target_ovx_mean']+=1
        with self.assertRaisesRegex(ValueError,'disagree'):e.common_test_weeks({'a':a,'b':other},'a')

    def test_bootstrap_blocks_preserve_calendar_dependence_and_are_deterministic(self):
        weeks=['2024-01-01','2024-01-08','2024-01-15','2024-02-05','2024-02-12','2024-02-19']
        indices=e.block_indices(weeks,2,100,42)
        np.testing.assert_array_equal(indices,e.block_indices(weeks,2,100,42))
        self.assertEqual(indices.shape,(100,6))
        self.assertTrue(np.all(indices[:,1::2]-indices[:,::2]==1))
        self.assertNotIn(2,indices[:,::2])
        for length in (0,7):
            with self.assertRaises(ValueError):e.block_indices(weeks,length,10,42)
        with self.assertRaises(ValueError):e.block_indices(weeks[::-1],2,10,42)

    def test_paired_bootstrap_constant_losses_and_full_block_identity(self):
        setting=policy()['bootstrap']
        paired=[{'target_week_start':r['target_week_start'],'absolute_error_M3_minus_M2':1.,
                 'squared_error_M2':4.,'squared_error_M3':9.} for r in sample() if r['split']=='test']
        result=e.bootstrap_interval(paired,2,setting)
        for key in ('mae_difference','rmse_difference'):
            self.assertEqual(result[key],{'point':1.,'lower':1.,'upper':1.})
        for i,r in enumerate(paired):
            r.update(absolute_error_M3_minus_M2=i-3,squared_error_M2=float(i*i),squared_error_M3=float((7-i)**2))
        result=e.bootstrap_interval(paired,8,setting)
        self.assertEqual(result['mae_difference'],{'point':.5,'lower':.5,'upper':.5})
        self.assertEqual(result['rmse_difference'],{'point':0.,'lower':0.,'upper':0.})

    def test_crisis_exclusion_uses_interval_overlap_without_refitting(self):
        p=policy()
        for start,end,expected in [('2023-10-23','2023-10-27',True),('2023-10-30','2023-11-03',False),
                                   ('2024-06-24','2024-06-28',False),('2024-07-01','2024-07-05',True)]:
            self.assertEqual(e.outside_crisis({'target_week_start':start,'target_week_end':end},p),expected)

    def test_full_compute_keeps_first_fits_and_all_sensitivities(self):
        p=policy();panels={k:sample(v['timing']=='conservative') for k,v in p['panels'].items()}
        p['crisis_exclusion'].update(start='2024-01-01',end='2024-01-07')
        snapshots={m:{**b.fit_ridge(sample()[:5],p['selected_alphas'][m],p|{'features':fields}),'model':m}
                   for m,fields in p['model_features'].items()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();(root/e.LOCK).write_text('{}')
            stage=root/'stage';stage.mkdir()
            with patch.object(e,'load_inputs',return_value=(panels,snapshots,{})):
                summary,_=e.compute(stage,root,p)
            self.assertEqual(summary['primary']['models']['M0']['weeks'],8)
            self.assertEqual(summary['crisis_exclusion']['models']['M3']['weeks'],7)
            self.assertTrue(summary['final_test_scored'])
            self.assertFalse(summary['retuned_from_final_test'])
            self.assertEqual(len(b.read(stage/'test_predictions.jsonl')),64)
            self.assertEqual(set(summary['bootstrap']),{'1','2','4'})
            self.assertEqual(set(summary['panels_common_weeks']),set(p['panels']))

    def fixture_root(self,root):
        p=policy();(root/'config').mkdir()
        (root/e.POLICY).write_text(json.dumps(p))
        b.save_json(root/e.LOCK,{'version':p['version'],'policy_sha256':b.digest(root/e.POLICY)})
        (root/'scripts').mkdir()
        for name in ('evaluate_finance.py','financial_baselines.py','financial_news.py'):
            (root/'scripts'/name).write_text('fixture')
        return p

    def test_policy_lock_rejected_before_any_final_rows_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.fixture_root(root)
            (root/e.POLICY).write_text('changed')
            with patch.object(b,'read',side_effect=AssertionError('Cannot open rows')):
                with self.assertRaisesRegex(ValueError,'pre-test lock'):e.load_inputs(root,p)

    def test_exposure_precedes_computation_and_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.fixture_root(root)
            def compute(stage,root,policy):
                exposure=root/(policy['output_directory']+'.exposure.json')
                self.assertTrue(exposure.is_file())
                b.save_json(stage/'summary.json',{'complete':True})
                return {'complete':True},{e.POLICY:b.digest(root/e.POLICY)}
            with patch.object(e,'compute',side_effect=compute):
                self.assertEqual(e.run(root),e.run(root,True))
                with self.assertRaisesRegex(ValueError,'exists'):e.run(root)
                (root/p['output_directory']/'summary.json').write_text('changed')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):e.run(root,True)

    def test_interrupted_evaluation_cannot_resume_after_implementation_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture_root(root)
            with patch.object(e,'compute',side_effect=ValueError('interruption')):
                with self.assertRaisesRegex(ValueError,'interruption'):e.run(root)
                (root/'scripts/evaluate_finance.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'changed after'):e.run(root)


if __name__=='__main__':
    unittest.main()
