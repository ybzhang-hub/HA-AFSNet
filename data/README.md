# External data

Use the released EG, IR and IS event data and LDA K5 labels. Set `HA_AFSNET_DATA_ROOT` to the following parent directory, or put this structure under the repository's `data/` directory:

```text
<data-root>/
├── data.zip                         # Optional, extracted by prepare_data.py
├── data_disentangled.zip             # Optional, extracted by prepare_data.py
├── data/{EG,IR,IS}/
│   ├── stat.txt                     # Entity and relation counts
│   ├── entity2id.txt / relation2id.txt
│   ├── head_ents.json               # Released popularity exclusions
│   ├── <region>.csv                 # Calendar dates and article links
│   ├── {train,valid,test}.txt        # s, r, o, timestamp
│   ├── {train,valid,test}_w_md5s.txt # Event-to-article links
│   ├── train_md5_list.json / md5_list.json
│   └── docs_title_paragraph.json    # Required for optional context fitting
└── data_disentangled/{EG,IR,IS}_LDA_K5/
    └── {train,valid,test}_w_contextid.txt  # s, r, o, timestamp, context
```

Preparation verifies the released files against `splits/<region>.json`; it does not construct a new random split. Exact article fitting identifiers and split date mappings are generated from the checked inputs. Raw articles, caches and trained weights are external inputs or generated outputs and are excluded from the source repository.
