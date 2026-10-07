# M2 duplicate retrieval (preliminary)

Preliminary retrieval check for duplicate detection; the full eval suite is built in M4.

- Model: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Store: in-memory Qdrant, whole synthetic dataset ingested (bugs, crashes, patch notes, docs)
- Query: duplicate's `title + description`; the query bug itself is excluded
- Cases: 16 bug reports with `duplicate_of`
- Command: `uv run python evals/duplicate_retrieval.py` (also run by the `slow` pytest test)

## Results

| Setting | Hits | Hit-rate@5 | Hit-rate@1 | MRR@5 |
|---|---|---|---|---|
| All doc types (primary) | 16/16 | 1.000 | 0.562 | 0.771 |
| `doc_types=[bug_report]` | 16/16 | 1.000 | 0.812 | 0.906 |

## Per case (all doc types)

| Duplicate | Original | Rank | Top-5 retrieved |
|---|---|---|---|
| BUG-0006 | BUG-0002 | 2 | BUG-0011, BUG-0002, DOC-coop-and-cross-save, BUG-0043, CRASH-0002 |
| BUG-0010 | BUG-0001 | 3 | PATCH-1.0.1, BUG-0015, BUG-0001, CRASH-0001, DOC-troubleshooting-guide |
| BUG-0011 | BUG-0002 | 2 | BUG-0006, BUG-0002, CRASH-0002, BUG-0043, PATCH-1.0.1 |
| BUG-0013 | BUG-0003 | 1 | BUG-0003, DOC-coop-and-cross-save, BUG-0045, BUG-0038, DOC-troubleshooting-guide |
| BUG-0015 | BUG-0001 | 1 | BUG-0001, BUG-0010, DOC-troubleshooting-guide, DOC-player-faq, BUG-0012 |
| BUG-0019 | BUG-0004 | 2 | PATCH-1.1.0, BUG-0004, PATCH-1.0.1, CRASH-0006, BUG-0021 |
| BUG-0023 | BUG-0008 | 1 | BUG-0008, DOC-troubleshooting-guide, CRASH-0007, DOC-system-requirements, BUG-0034 |
| BUG-0028 | BUG-0018 | 2 | DOC-troubleshooting-guide, BUG-0018, CRASH-0009, PATCH-1.1.1, PATCH-1.1.1 |
| BUG-0030 | BUG-0027 | 2 | DOC-troubleshooting-guide, BUG-0027, CRASH-0008, PATCH-1.1.1, BUG-0050 |
| BUG-0039 | BUG-0034 | 1 | BUG-0034, DOC-troubleshooting-guide, DOC-troubleshooting-guide, CRASH-0012, CRASH-0017 |
| BUG-0044 | BUG-0042 | 1 | BUG-0042, BUG-0049, PATCH-1.2.1, DOC-coop-and-cross-save, DOC-player-faq |
| BUG-0045 | BUG-0038 | 1 | BUG-0038, DOC-coop-and-cross-save, BUG-0013, PATCH-1.2.1, DOC-troubleshooting-guide |
| BUG-0049 | BUG-0042 | 1 | BUG-0042, PATCH-1.2.1, BUG-0044, DOC-coop-and-cross-save, CRASH-0014 |
| BUG-0052 | BUG-0048 | 2 | DOC-troubleshooting-guide, BUG-0048, PATCH-1.3.1, CRASH-0016, BUG-0043 |
| BUG-0055 | BUG-0050 | 1 | BUG-0050, PATCH-1.3.0, DOC-coop-and-cross-save, DOC-system-requirements, DOC-troubleshooting-guide |
| BUG-0060 | BUG-0059 | 1 | BUG-0059, PATCH-1.4.0, PATCH-1.4.1, PATCH-1.4.1, CRASH-0021 |

## Misses

- none
