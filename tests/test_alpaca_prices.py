import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca_prices import AlpacaPrices, assess_prices, completed_sessions, load_ranking_prices


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class Session:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, dict(kwargs['params'])))
        return next(self.responses)


class PriceTests(unittest.TestCase):
    def test_calendar_completed_session(self):
        cases = {
            '2026-09-23T01:17Z': '2026-09-22',
            '2026-09-07T22:00Z': '2026-09-04',  # Labor Day
            '2026-11-27T18:29Z': '2026-11-25',  # early close plus delay
            '2026-11-27T18:31Z': '2026-11-27',
            '2026-03-09T20:31Z': '2026-03-09',  # DST
        }
        for now, expected in cases.items():
            with self.subTest(now=now):
                sessions, end = completed_sessions(now)
                self.assertEqual(str(sessions[-1].date()), expected)
                self.assertLessEqual(end, pd.Timestamp(now) - pd.Timedelta(minutes=15))
                self.assertEqual(str(end.tz_convert('America/New_York').date()), expected)

    def test_pagination_class_symbols_and_ny_dates(self):
        session = Session([
            Response({'bars': {'AAPL': [{'t': '2026-09-22T04:00:00Z', 'c': 100}]},
                      'next_page_token': 'page2'}),
            Response({'bars': {'BRK.B': [{'t': '2026-09-22T04:00:00Z', 'c': 500}]},
                      'next_page_token': None}),
        ])
        client = AlpacaPrices('test-key', 'test-secret', session=session)
        prices = client.closes(['AAPL', 'BRK-B'], '2026-09-21T04:00Z', '2026-09-23T03:59Z')
        self.assertEqual(prices.loc['2026-09-22', 'BRK-B'], 500)
        self.assertEqual(session.calls[0][1]['symbols'], 'AAPL,BRK.B')
        self.assertEqual(session.calls[1][1]['page_token'], 'page2')
        self.assertEqual(session.calls[0][1]['adjustment'], 'all')
        self.assertEqual(session.calls[0][1]['feed'], 'sip')

    def test_no_fill_and_distinct_exclusion_reasons(self):
        dates = pd.bdate_range('2026-01-01', periods=130)
        tickers = ['OK', 'OLD_GAP', 'HOLE', 'IPO', 'STALE', 'RAW_MISSING']
        prices = pd.DataFrame(100., index=dates, columns=tickers)
        prices.loc[dates[0], 'OLD_GAP'] = np.nan
        prices.loc[dates[-20], 'HOLE'] = np.nan
        prices.loc[dates[:50], 'IPO'] = np.nan
        prices.loc[dates[-1], 'STALE'] = np.nan
        raw = pd.Series(110., index=tickers)
        raw['RAW_MISSING'] = np.nan
        data, eligible, quality = assess_prices(prices, raw, dates, tickers, 125)
        details = {x['ticker']: x for x in quality['symbols']}
        self.assertEqual(eligible, ['OK', 'OLD_GAP'])
        self.assertEqual(details['HOLE']['reasons'], ['missing_in_lookback'])
        self.assertEqual(details['IPO']['reasons'], ['insufficient_history'])
        self.assertEqual(details['IPO']['missing_dates'], [])
        self.assertIn('missing_latest', details['STALE']['reasons'])
        self.assertIn('missing_latest', details['RAW_MISSING']['reasons'])
        self.assertTrue(pd.isna(data.loc[dates[-20], 'HOLE']))
        self.assertEqual(details['OK']['raw_close'], 110.)
        self.assertEqual(data.loc[dates[-1], 'OK'], 100.)
        json.dumps(quality, allow_nan=False)

    def test_retry_replaces_adjustment_vintage_and_restores_missing_last_row(self):
        dates, _ = completed_sessions('2026-09-23T01:17Z')
        class Client:
            def __init__(self):
                self.calls = 0
            def closes(self, tickers, start, end, adjustment='all'):
                self.calls += 1
                if adjustment == 'raw':
                    return pd.DataFrame(250., index=dates[-1:], columns=tickers)
                if self.calls == 1:
                    return pd.DataFrame(100., index=dates[:-1], columns=tickers)
                return pd.DataFrame(200., index=dates, columns=tickers)
        with tempfile.TemporaryDirectory() as tmp:
            data, raw, eligible, q = load_ranking_prices(
                ['A'], client=Client(), now='2026-09-23T01:17Z',
                report_path=Path(tmp)/'quality.json', log=lambda _: None)
        self.assertEqual(eligible, ['A'])
        self.assertTrue(data['A'].eq(200).all())
        self.assertEqual(raw['A'], 250)
        self.assertEqual(q['filled_cells'], 0)

    def test_whole_feed_stale_refuses_snapshot_but_writes_report(self):
        dates, _ = completed_sessions('2026-09-23T01:17Z')
        class Client:
            def closes(self, tickers, start, end, adjustment='all'):
                return pd.DataFrame(100., index=dates[:-1], columns=tickers)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'quality.json'
            with self.assertRaisesRegex(RuntimeError, '95%'):
                load_ranking_prices(['A'], client=Client(), now='2026-09-23T01:17Z',
                                    report_path=path, log=lambda _: None)
            self.assertEqual(json.loads(path.read_text())['fresh'], 0)

    def test_auth_failure_does_not_expose_credentials(self):
        session = Session([Response({'message': 'secret-value'}, status=403)])
        client = AlpacaPrices('key-value', 'secret-value', session=session)
        with self.assertRaises(RuntimeError) as caught:
            client.closes(['A'], '2026-09-21T04:00Z', '2026-09-23T03:59Z')
        self.assertNotIn('secret-value', str(caught.exception))
        self.assertEqual(len(session.calls), 1)


if __name__ == '__main__':
    unittest.main()
