"""Synthetic exports test data-integrity cases without copying licensed text."""

from pathlib import Path
import tempfile
import unittest

from scripts.import_proquest import import_files, parse_record


def example(document_id="123", body="Ships resumed operations.\nSecond paragraph.",
            published="Jan 31, 2019", extra="", title="Example report"):
    return (f"Texte intégral: {body}\n\n"
            "Terme d'indexation pour les affaires: Shipping\n\n"
            f"Titre: {title}\n\nPublication: Example News\n\n"
            f"Date de publication: {published}\n\n"
            f"ID de document ProQuest: {document_id}\n\n{extra}"
            "Dernière mise à jour: 2020-01-01\n\nBase de données: Example\n")


class ImportTests(unittest.TestCase):
    def test_body_boundary_and_unknown_availability(self):
        r = parse_record(example())
        self.assertEqual(r["body_raw"], "Ships resumed operations.\nSecond paragraph.")
        self.assertNotIn("Shipping", r["body_clean"])
        self.assertEqual(r["publication_date"], "2019-01-31")
        self.assertIsNone(r["authors"])
        self.assertIsNone(r["archive_available_at"])
        self.assertEqual(r["date_precision"], "unverified")
        self.assertIn("month_end_date_review", r["quality_flags"])

    def test_delay_preserved_without_inventing_timestamp(self):
        r = parse_record(example(extra="Disponibilité du texte intégral: Content is not loaded until 30 days after it is published.\n\n"))
        self.assertIn("archive_loading_delay_30_days", r["quality_flags"])
        self.assertIsNone(r["archive_available_at"])

    def test_invalid_and_partial_dates_not_filled(self):
        for value in ("Feb 31, 2019", "Jan 2019", ""):
            with self.subTest(value=value):
                r = parse_record(example(published=value))
                self.assertIsNone(r["publication_date"])
                self.assertIn("publication_date_unparsed", r["quality_flags"])

    def test_repeated_exports_and_distinct_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"export{i}.txt" for i in range(3)]
            paths[0].write_bytes(("\ufeff" + example()).replace("\n", "\r\n").encode())
            paths[1].write_text(example(), encoding="utf-8")
            paths[2].write_text(example(body="A revised report.") + "\n" + "_" * 60 + "\n" + example(document_id="456"), encoding="utf-8")
            records, report = import_files(paths)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["parsed_occurrences"], 4)
            self.assertEqual(report["unique_ids"], 2)
            self.assertEqual(report["record_versions"], 3)
            self.assertEqual(report["repeated_identical_versions"], 1)
            self.assertEqual(len(report["ids_with_multiple_versions"]["123"]), 2)
            self.assertEqual(list(report["identical_bodies_across_ids"].values()), [["123", "456"]])
            self.assertEqual(sum(len(r["provenance"]) for r in records), 4)

    def test_truncated_record_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.txt"
            path.write_text("Texte intégral: Truncated report\n", encoding="utf-8")
            records, report = import_files([path])
            self.assertEqual(records, [])
            self.assertEqual(len(report["errors"]), 1)

    def test_spaced_id_requires_url_corroboration(self):
        record = parse_record(example(document_id="12 3", extra="URL du document: https://www.proquest.com/docview/123/sem-2\n\n"))
        self.assertEqual(record["document_id"], "123")
        self.assertEqual(record["document_id_raw"], "12 3")
        self.assertEqual(record["metadata_raw"]["ID de document ProQuest"], "12 3")
        self.assertIn("document_id_spacing_normalized_url_verified", record["quality_flags"])
        for raw_id, url in (("12 3", ""), ("12 3", "https://www.proquest.com/docview/456"),
                            ("12x3", "https://www.proquest.com/docview/123")):
            with self.subTest(raw_id=raw_id, url=url), self.assertRaisesRegex(ValueError, "Malformed document ID"):
                parse_record(example(document_id=raw_id, extra=f"URL du document: {url}\n\n"))

    def test_date_label_variant_preserves_metadata(self):
        record = parse_record(example().replace("Date de publication:", "Datede publication:"))
        self.assertEqual(record["publication_date"], "2019-01-31")
        self.assertEqual(record["metadata_raw"]["Datede publication"], "Jan 31, 2019")
        self.assertIn("publication_date_label_variant", record["quality_flags"])

    def test_id_label_variant_preserves_metadata(self):
        record = parse_record(example().replace("ID de document ProQuest:", "IDde document ProQuest:"))
        self.assertEqual(record["document_id"], "123")
        self.assertEqual(record["metadata_raw"]["IDde document ProQuest"], "123")
        self.assertIn("document_id_label_variant", record["quality_flags"])

    def test_spaced_date_requires_corresponding_copyright_date(self):
        for copyright_date, expected in (("May 27, 2024", "2024-05-27"), ("May 28, 2024", None), ("", None)):
            with self.subTest(copyright_date=copyright_date):
                record = parse_record(example(published="May 2 7, 2024", extra=f"Copyright: Copyright Example {copyright_date}\n\n"))
                self.assertEqual(record["publication_date"], expected)
                self.assertEqual(record["publication_date_raw"], "May 2 7, 2024")

    def test_known_publication_and_availability_spacing(self):
        record = parse_record(example(extra="Disponibilité du texte intégral: Content is not loaded until 30 days a fter it is published.\n\n").replace("Publication: Example News", "Publication: Financi al Times; Lond on (UK)"))
        self.assertEqual(record["publication_group"], "Financial Times; London (UK)")
        self.assertEqual(record["publication_raw"], "Financi al Times; Lond on (UK)")
        self.assertIn("publication_name_spacing_normalized", record["quality_flags"])
        self.assertIn("archive_loading_delay_30_days", record["quality_flags"])
        self.assertIsNone(parse_record(example())["publication_group"])


if __name__ == "__main__":
    unittest.main()
