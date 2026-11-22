## Retrieval (overall)

|            |   Recall@1 |   Recall@3 |   Recall@5 |   Recall@10 |   MRR@10 |   nDCG@5 |
|:-----------|-----------:|-----------:|-----------:|------------:|---------:|---------:|
| colpali    |      0.867 |      0.967 |      0.967 |       1     |    0.91  |    0.921 |
| ocr-bm25   |      0.767 |      0.9   |      0.933 |       0.967 |    0.836 |    0.856 |
| ocr-dense  |      0.367 |      0.733 |      0.933 |       0.967 |    0.576 |    0.661 |
| ocr-hybrid |      0.467 |      0.9   |      0.933 |       1     |    0.697 |    0.753 |

## Recall@1 by modality

|             |   n |   colpali |   ocr-bm25 |   ocr-dense |   ocr-hybrid |
|:------------|----:|----------:|-----------:|------------:|-------------:|
| chart       |  15 |     0.867 |      0.8   |       0.467 |        0.533 |
| diagram     |   2 |     1     |      0.5   |       0     |        0     |
| infographic |   1 |     1     |      1     |       0     |        0     |
| map         |   2 |     1     |      1     |       0.5   |        0.5   |
| table       |   7 |     0.857 |      0.571 |       0.286 |        0.429 |
| text        |   3 |     0.667 |      1     |       0.333 |        0.667 |

## Paired comparison vs ColPali

|            |   ColPali wins @1 |   text wins @1 |   ties |   sign-test p |   dRecall@1 | dRecall@1 95% CI                        |   dMRR@10 | dMRR@10 95% CI                          |
|:-----------|------------------:|---------------:|-------:|--------------:|------------:|:----------------------------------------|----------:|:----------------------------------------|
| ocr-bm25   |                 5 |              2 |     23 |        0.4531 |         0.1 | (np.float64(-0.067), np.float64(0.267)) |     0.074 | (np.float64(-0.046), np.float64(0.202)) |
| ocr-dense  |                16 |              1 |     13 |        0.0003 |         0.5 | (np.float64(0.3), np.float64(0.7))      |     0.334 | (np.float64(0.203), np.float64(0.463))  |
| ocr-hybrid |                14 |              2 |     14 |        0.0042 |         0.4 | (np.float64(0.167), np.float64(0.6))    |     0.213 | (np.float64(0.078), np.float64(0.342))  |

## Efficiency

|    | Arm              | Indexing / page                         | Storage / page               | Query: encode   | Query: search              | Query: encode+search   |
|---:|:-----------------|:----------------------------------------|:-----------------------------|:----------------|:---------------------------|:-----------------------|
|  0 | ColPali + MaxSim | not measured (paper: 0.39 s on its GPU) | 258 KB (patch matrix)        | 17.2 s (cpu)    | 344 ms (exhaustive MaxSim) |                        |
|  1 | OCR + BM25       | 20.1 s OCR (CPU)                        | 3.4 KB (text)                | none            | 6 ms                       |                        |
|  2 | OCR + dense      | 20.1 s OCR + embedding                  | 3.4 KB text + 6.5 KB vectors |                 |                            | 130 ms (cpu)           |
