import unittest

import pandas as pd

from futures_curves import curve_at, validate


class FuturesCurveTests(unittest.TestCase):
    def test_same_session_curve_uses_only_visible_contract_vintages(self):
        common = {"commodity": "wti", "session": "2024-01-03", "currency": "USD",
                  "price_unit": "barrel", "source": "fixture"}
        frame = validate(pd.DataFrame([
            {**common, "contract_id": sid, "expiry": expiry, "settlement": value,
             "available_at_utc": f"2024-01-03T{hour}:00:00Z"}
            for sid, expiry, value, hour in [
                ("A", "2024-02-01", 75, "20"),
                ("B", "2024-03-01", 70, "20"),
                ("A", "2024-02-01", 65, "22"),
            ]
        ]))
        early = curve_at(frame, "2024-01-03T21:00Z", "2024-01-03")["wti"]
        late = curve_at(frame, "2024-01-03T23:00Z", "2024-01-03")["wti"]
        self.assertEqual(early["spread_front_minus_second"], 5)
        self.assertEqual(late["shape"], "contango")
        self.assertEqual(curve_at(frame, "2024-01-04T23:00Z", "2024-01-04"), {})

    def test_settlement_cannot_be_available_before_its_session(self):
        record = {"commodity": "wti", "session": "2024-01-03", "expiry": "2024-02-01",
                  "currency": "USD", "price_unit": "barrel", "source": "fixture",
                  "contract_id": "A", "settlement": 75, "available_at_utc": "2024-01-03T02:00:00Z"}
        with self.assertRaisesRegex(ValueError, "precedes its session"):
            validate(pd.DataFrame([record]))
