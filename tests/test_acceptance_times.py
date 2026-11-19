import unittest

import pandas as pd

from data_8k import accepted_et


class AcceptedTimeTests(unittest.TestCase):
    """EDGAR index 'Accepted' times equal the submissions API value converted to New York time (checked 2026-11-19)."""

    def test_api_time_is_converted_without_offset_shift(self):
        # 0001193125-18-204064: API 2018-06-26T20:15:59Z, EDGAR index Accepted 2018-06-26 16:15:59 (EDT)
        out = accepted_et(pd.Series(["2018-06-26T20:15:59.000Z"]))
        self.assertEqual(out.iloc[0], pd.Timestamp("2018-06-26 16:15:59", tz="America/New_York"))

    def test_winter_offset(self):
        # 0001140361-26-002064: API 2026-01-22T22:00:30Z, EDGAR index Accepted 2026-01-22 17:00:30 (EST)
        out = accepted_et(pd.Series(["2026-01-22T22:00:30.000Z"]))
        self.assertEqual(out.iloc[0], pd.Timestamp("2026-01-22 17:00:30", tz="America/New_York"))

    def test_after_close_filing_is_not_moved_into_the_session(self):
        # a 16:05 ET earnings release must stay after the 16:00 close (next-session availability)
        out = accepted_et(pd.Series(["2024-02-01T21:05:00Z"]))
        self.assertGreaterEqual(out.iloc[0].hour, 16)

    def test_missing_values_stay_missing(self):
        self.assertTrue(accepted_et(pd.Series([None, "bad"])).isna().all())


class IndexPageTests(unittest.TestCase):
    PAGE = ('<div class="formGrouping"><div class="infoHead">Accepted</div>\n'
            '<div class="info">2019-08-07 16:23:29</div></div>')

    def test_index_accepted_parsed_as_new_york_time(self):
        from insider_acceptances import parse_index_accepted
        self.assertEqual(parse_index_accepted(self.PAGE), "2019-08-07T20:23:29+00:00")
        winter = self.PAGE.replace("2019-08-07 16:23:29", "2020-01-28 19:31:26")
        self.assertEqual(parse_index_accepted(winter), "2020-01-29T00:31:26+00:00")
        self.assertIsNone(parse_index_accepted("<html>no field</html>"))

    def test_missing_filing_returns_none_without_inferring(self):
        from insider_acceptances import index_acceptance

        class Missing:
            def get(self, url):
                return type("R", (), {"status_code": 404, "text": ""})()
        self.assertIsNone(index_acceptance(Missing(), [1], "000122520825003519", pause=lambda s: None, attempts=1))
        with self.assertRaises(ValueError):
            index_acceptance(Missing(), [1], "bad", pause=lambda s: None)


if __name__ == "__main__":
    unittest.main()
