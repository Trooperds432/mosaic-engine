"""Authoritative input/source registry for panel provenance.

Register each new feature data source here, including its original archive and
the transform version. Outputs (panels, audits, forecasts) are not input data.
"""
import config as C

REGISTRY_VERSION = 1
DEPENDENCIES = {
    'prices': {'raw': ['prices/*.parquet'], 'sources': ['data_prices.py']},
    'universe': {'proc': ['universe.parquet', 'sic.parquet', 'historical_bundle.json'],
                 'raw': ['historical/**/*.parquet', 'historical/**/coverage.json'],
                 'sources': ['data_universe.py', 'historical_data.py']},
    'fundamentals': {'raw': ['facts/companyfacts.parquet'], 'sources': ['data_fundamentals.py', 'data_updates.py']},
    'insiders': {'raw': ['insider/*.parquet', 'insider_recent/form4_recent.parquet'],
                 'sources': ['data_insider.py', 'data_form4_recent.py']},
    'institutions': {'proc': ['inst_ownership_cusip.parquet', 'cusip_cik.parquet'],
                     'raw': ['13f/names/*.parquet'], 'sources': ['data_13f.py']},
    'macro': {'proc': ['macro_pit.parquet', 'macro_surprises.parquet'],
              'raw': ['alt/**/*.parquet', 'macro/**/*.parquet', 'macro/**/*.csv'],
              'sources': ['data_macro.py', 'macro_surprise.py', 'data_alt.py', 'pit_clinical.py']},
    'events': {'proc': ['8k_items.parquet', 'fda_approvals.parquet', 'short_interest.parquet'],
               'sources': ['data_8k.py', 'data_alt.py']},
    'estimates': {'proc': ['av_earnings.parquet', 'finnhub_recommendations.parquet', 'estimate_vintages.parquet'],
                  'raw': ['collect/**/*.parquet', 'estimate_vintages/*.csv'],
                  'sources': ['data_collect.py', 'pit_estimates.py', 'provider_history.py']},
    'filing_text': {'proc': ['text_similarity.parquet'], 'raw': ['text/**/*.parquet', 'text/**/*.txt.gz'],
                    'sources': ['data_text.py']},
    'feature_transform': {'sources': ['features.py', 'config.py', 'provenance.py', 'feature_dependencies.py']},
}


def input_paths():
    """Return data/source paths plus explicit missing-file markers.

    Both absent fixed inputs and empty archive patterns have identities; adding
    or deleting a dependency therefore changes the dataset fingerprint.
    """
    paths = set()
    for group in DEPENDENCIES.values():
        for kind, patterns in group.items():
            base = {'raw': C.RAW, 'proc': C.PROC, 'sources': C.ROOT}[kind]
            for pattern in patterns:
                if '*' in pattern:
                    paths.update(p for p in base.glob(pattern) if p.is_file())
                else:
                    paths.add(base / pattern)
    return sorted(paths)
