import copy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_weekly_finance as w


def policy():
    return json.loads((w.ROOT / w.POLICY).read_text())


def article(doc="a", source=w.GUARDIAN, day="2023-01-04", title="Red Sea shipping threatened"):
    body = "Red Sea shipping faces military threats."
    hashed = w.text_digest(body)
    return dict(source=source, document_id=doc, record_version_hash="version-" + doc,
        version_hashes=["version-" + doc], body_clean=body, body_hash=hashed, title=title,
        publication_date=day, exact_group_id=w.text_digest(source + "\n" + hashed),
        near_family_id="family-" + doc)


def assignment(d):
    return {**d, "split_group_id": "component-" + d["document_id"]}


def predicted(doc="a", source=w.GUARDIAN, day="2023-01-04", group="body-a", relevant=1):
    return dict(source=source, document_id=doc, record_version_hash="version-" + doc,
        exact_group_id=group, near_family_id=group, split_group_id="component-" + group,
        publication_proxy_at=day + "T23:59:59+00:00", source_url="https://example.com/" + doc,
        predictions=dict(relevant=relevant, threat=relevant, disruption=0))


def small_policy():
    p = policy()
    p.update(start="2023-01-02", end="2023-02-26")
    return p


def audits():
    return {2023:"counts_reconciled"}, {"2023-01", "2023-02", "2023-03"}


class WeeklyFinanceTests(unittest.TestCase):
    def test_classifier_uses_frozen_full_text_and_not_annotation_labels(self):
        d = article()
        d["labels"] = dict(relevant=0, threat=0, disruption=0)
        config = json.loads((w.ROOT / "config/nlp_comparison_v1.json").read_text())
        original = copy.deepcopy(d)
        preds, excluded = w.classify_documents([d], [assignment(d)], {"quality_decisions":[]}, config["keyword"], policy())
        self.assertEqual(preds[0]["predictions"], dict(relevant=1, threat=1, disruption=0))
        self.assertEqual(preds[0]["input_sha256"], w.text_digest(d["title"] + "\n\n" + d["body_clean"]))
        self.assertEqual(d, original)
        self.assertEqual(excluded, [])

    def test_quality_exclusions_cleanup_and_stale_decision(self):
        ds = [article("clean"), article("exclude"), article("special", source="Other")]
        ds[0]["body_clean"] += " Sujet ERIC: index"
        ds[0]["body_hash"] = w.text_digest(ds[0]["body_clean"])
        ds[0]["exact_group_id"] = w.text_digest(ds[0]["source"] + "\n" + ds[0]["body_hash"])
        decisions=[]
        for d, action in zip(ds, ["remove_metadata_suffix", "exclude"]):
            decisions.append({**{k:d[k] for k in ("source", "document_id", "record_version_hash", "body_hash")},
                "action":action, "flags":[], "suffix_start":len("Red Sea shipping faces military threats."),
                "original_length":len(d["body_clean"])})
        config=json.loads((w.ROOT / "config/nlp_comparison_v1.json").read_text())
        preds, excluded=w.classify_documents(ds, list(map(assignment,ds)), {"quality_decisions":decisions}, config["keyword"], policy())
        self.assertEqual(len(preds),1)
        self.assertEqual(preds[0]["body_model_sha256"], w.text_digest("Red Sea shipping faces military threats."))
        self.assertEqual({r["reason"] for r in excluded},{"development_quality_exclusion","supplementary_source"})
        decisions[0]["body_hash"]="changed"
        with self.assertRaisesRegex(ValueError,"Stale"):
            w.classify_documents(ds,list(map(assignment,ds)),{"quality_decisions":decisions},config["keyword"],policy())

    def test_duplicate_identity_wrong_assignment_and_body_tamper_rejected(self):
        d=article()
        with self.assertRaisesRegex(ValueError,"identities"):
            w.classify_documents([d,d],[assignment(d)],{"quality_decisions":[]},{},policy())
        a=assignment(d);a["publication_date"]="2024-01-04"
        with self.assertRaisesRegex(ValueError,"mismatch"):
            w.classify_documents([d],[a],{"quality_decisions":[]},{},policy())
        a=assignment(d);d["body_clean"]+="changed"
        with self.assertRaisesRegex(ValueError,"body hash"):
            w.classify_documents([d],[a],{"quality_decisions":[]},{},policy())

    def test_timestamp_timezone_and_date_only_assumptions(self):
        d=article(day="2023-01-08")
        t,basis=w.timestamp(d)
        self.assertEqual(t.hour,23)
        self.assertIn("assumption",basis)
        d["publication_datetime"]="2023-01-09T00:30:00+01:00"
        self.assertEqual(w.timestamp(d)[0].date(),date(2023,1,8))
        d["publication_datetime"]="2023-01-09T00:30:00"
        with self.assertRaisesRegex(ValueError,"timezone"):
            w.timestamp(d)
        d["publication_datetime"]="2023-01-09T00:30:00Z"
        with self.assertRaisesRegex(ValueError,"disagree"):
            w.timestamp(d)

    def test_exact_copies_do_not_import_later_headlines_and_sources_stay_separate(self):
        rows=[predicted("first",day="2023-01-03",relevant=0),
              predicted("later",day="2023-01-06",relevant=1),
              predicted("next-week",day="2023-01-10",relevant=1),
              predicted("ft",source=w.FT,day="2023-01-04",group="ft-body")]
        news,members,_=w.aggregate_news(rows,small_policy(),*audits())
        r=next(r for r in news if (r["timing"],r["view"],r["week_start"])==("publication","combined","2023-01-02"))
        self.assertEqual((r["candidate_count"],r["counts"]["relevant"],r["exact_copies_removed"]),(2,1,1))
        self.assertEqual(r["raw_shares"]["threat"],0.5)
        self.assertTrue(any(m["document_id"]=="later" and not m["counted"] and m["disagrees_with_representative"] for m in members))
        later=next(r for r in news if (r["timing"],r["view"],r["week_start"])==("publication","guardian","2023-01-09"))
        self.assertEqual(later["counts"]["relevant"],1)
        self.assertEqual(news,w.aggregate_news(list(reversed(rows)),small_policy(),*audits())[0])

    def test_sunday_cutoff_and_predeclared_delays(self):
        rows=[predicted("sun",day="2023-01-08"),predicted("mon",day="2023-01-09"),
              predicted("ft",source=w.FT,day="2023-01-08",group="ft")]
        _,members,_=w.aggregate_news(rows,small_policy(),*audits())
        mapped={(r["timing"],r["document_id"]):r["week_start"] for r in members}
        self.assertEqual(mapped["publication","sun"],"2023-01-02")
        self.assertEqual(mapped["publication","mon"],"2023-01-09")
        self.assertEqual(mapped["conservative","sun"],"2023-01-09")
        self.assertEqual(mapped["conservative","ft"],"2023-02-06")

    def test_zero_denominator_and_unknown_coverage_are_distinct(self):
        news,_,_=w.aggregate_news([],small_policy(),*audits())
        g=next(r for r in news if (r["timing"],r["view"],r["week_start"])==("publication","guardian","2023-01-09"))
        ft=next(r for r in news if (r["timing"],r["view"],r["week_start"])==("publication","ft","2023-01-09"))
        self.assertIsNone(g["raw_shares"]["threat"])
        self.assertEqual(g["features"]["threat_share"],0)
        self.assertEqual(g["zero_interpretation"],"no_eligible_query_matches")
        self.assertEqual(ft["zero_interpretation"],"no_observed_records_completeness_unverified")
        unknown,_,_=w.aggregate_news([],small_policy(),{},set())
        self.assertTrue(all(not r["news_features_available"] and r["features"]["threat_share"] is None for r in unknown))

    def test_accepted_ft_gap_and_lagged_boundary_are_flagged(self):
        news,_,_=w.aggregate_news([],small_policy(),{2023:"partial_pending_search_reconciliation"},audits()[1])
        r=next(r for r in news if (r["timing"],r["view"],r["week_start"])==("publication","ft","2023-01-09"))
        self.assertTrue(r["news_features_available"])
        self.assertEqual(r["coverage"][0]["status"],"known_annual_gap_accepted")
        r=next(r for r in news if (r["timing"],r["view"],r["week_start"])==("conservative","ft","2023-01-09"))
        self.assertFalse(r["news_features_available"])

    def test_shifted_articles_outside_range_are_accounted_for(self):
        news,members,overflow=w.aggregate_news([predicted(day="2023-02-26")],small_policy(),*audits())
        self.assertEqual(len(overflow),1)
        self.assertEqual(overflow[0]["timing"],"conservative")
        self.assertEqual(len(members),1)

    def test_ovx_blanks_absent_days_and_partial_weeks_are_not_filled(self):
        p=small_policy();p.update(start="2023-01-03",end="2023-02-22")
        values={date(2023,1,9):10,date(2023,1,10):20,date(2023,1,11):30,date(2023,1,12):None}
        rows=w.aggregate_ovx(values,p)
        r=next(r for r in rows if r["week_start"]=="2023-01-09")
        self.assertEqual(r["observed_mean"],20)
        self.assertEqual(r["explicit_missing_dates"],["2023-01-12"])
        self.assertEqual(r["absent_weekdays"],["2023-01-13"])
        self.assertTrue(r["eligible"])
        self.assertFalse(rows[0]["eligible"])
        self.assertFalse(rows[-1]["eligible"])
        self.assertEqual(r["imputed_observations"],0)

    def test_ovx_rejects_invalid_values_dates_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"ovx.csv"
            for body in ["2023-01-09,nan\n", "2023-01-09,0\n", "2023-01-08,10\n", "2023-01-09,10\n2023-01-09,20\n", "2023-01-10,10\n2023-01-09,20\n"]:
                path.write_text("observation_date,OVXCLS\n"+body)
                with self.assertRaises(ValueError): w.read_ovx(path,small_policy())
            path.write_text("observation_date,OVXCLS\n2023-01-09,10\n2023-01-10,.\n")
            self.assertEqual(w.read_ovx(path,small_policy()),{date(2023,1,9):10,date(2023,1,10):None})

    def test_target_alignment_and_future_outcomes_cannot_change_features(self):
        p=small_policy()
        news,_,_=w.aggregate_news([],p,*audits())
        values={d:float((d-date(2023,1,2)).days//7+1) for d in w.dates(date(2023,1,2),date(2023,2,24)) if d.weekday()<5}
        market=w.aggregate_ovx(values,p)
        rows=w.align_financial(news,market,p)
        r=next(r for r in rows if (r["timing"],r["view"],r["feature_week_start"])==("publication","combined","2023-01-23"))
        self.assertEqual(r["features"]["ovx_latest"],4)
        self.assertEqual(r["features"]["ovx_trailing_four_week_mean"],2.5)
        self.assertEqual(r["target_ovx_mean"],5)
        self.assertEqual(r["target_week_end"],"2023-02-03")
        self.assertTrue(r["eligible"])
        delayed=next(r for r in rows if (r["timing"],r["view"],r["feature_week_start"])==("conservative","guardian","2023-01-30"))
        self.assertEqual(delayed["features"]["ovx_latest"],4)
        self.assertEqual(delayed["target_ovx_mean"],6)
        values.update({d:999 for d in values if d>=date(2023,1,30)})
        changed=w.align_financial(news,w.aggregate_ovx(values,p),p)
        actual=next(x for x in changed if (x["timing"],x["view"],x["feature_week_start"])==("publication","combined","2023-01-23"))
        self.assertEqual(actual["features"],r["features"])
        self.assertNotEqual(actual["target_ovx_mean"],r["target_ovx_mean"])

    def test_target_crossing_split_boundary_is_never_training(self):
        p=policy()
        self.assertEqual(w.split_for_target(date(2023,12,25),date(2023,12,29),p),"validation")
        self.assertEqual(w.split_for_target(date(2024,1,1),date(2024,1,5),p),"test")
        self.assertIsNone(w.split_for_target(date(2022,12,30),date(2023,1,3),p))
        self.assertIsNone(w.split_for_target(date(2025,12,29),date(2026,1,2),p))

    def test_snapshot_refuses_overwrite_reproduces_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"config").mkdir()
            p=policy();(root/w.POLICY).write_text(json.dumps(p))
            def compute(stage,root,policy):
                w.save_json(stage/"summary.json",{"complete":True})
                return {"complete":True},{w.POLICY:w.digest(root/w.POLICY)}
            with patch.object(w,"compute",side_effect=compute):
                self.assertEqual(w.run(root),w.run(root,True))
                with self.assertRaisesRegex(ValueError,"exists"):w.run(root)
                (root/p["output_directory"]/"summary.json").write_text("changed")
                with self.assertRaisesRegex(ValueError,"hash mismatch"):w.run(root,True)


if __name__ == "__main__":
    unittest.main()
