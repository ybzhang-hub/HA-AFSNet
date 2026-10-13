"""Fit article-context models on training articles and assign heldout contexts."""

import argparse
import csv
import hashlib
import json
import os
import pickle
import string
import sys
from pathlib import Path

import nltk
import numpy as np
from gensim import corpora
from gensim.models import Doc2Vec, LdaModel
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ha_afsnet.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
)
from ha_afsnet.config import (
    PROJECT_ROOT as ROOT,
)
from ha_afsnet.utils import dump, sha256


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def clean_documents(articles):
    stop = set(stopwords.words("english"))
    excluded = set(string.punctuation) | {"'s", "``", "''"}
    lemma = WordNetLemmatizer()
    for title, paragraphs in articles:
        sentences = [title]
        for paragraph in paragraphs:
            sentences.extend(nltk.sent_tokenize(paragraph))
        yield [
            lemma.lemmatize(word)
            for sentence in sentences
            for word in sentence.split()
            if word not in stop
            and word not in excluded
            and all(ord(c) <= 127 for c in word)
        ]


def fitted_arrays(model, method):
    if method == "lda":
        return [model.state.sstats, model.expElogbeta]
    if method == "doc2vec":
        return [model.wv.vectors, model.dv.vectors, model.syn1neg]
    if method == "kmeans":
        return [model.cluster_centers_]
    return [
        model.weights_,
        model.means_,
        model.covariances_,
        model.precisions_cholesky_,
    ]


def fit_region(
    region, config, methods, output_root=None, data_root=None, prepared_root=None
):
    source = (data_root or DEFAULT_DATA_ROOT) / "data" / region
    evidence = (prepared_root or DEFAULT_PREPARED_ROOT) / region
    output = (output_root or ROOT / "outputs/contexts") / region
    output.mkdir(parents=True, exist_ok=True)
    md5s = json.loads((source / "md5_list.json").read_text())
    training_ids = json.loads((evidence / "train_article_md5s.json").read_text())
    valid_ids = json.loads((evidence / "valid_article_md5s.json").read_text())
    test_ids = json.loads((evidence / "test_article_md5s.json").read_text())
    if set(training_ids) & (set(valid_ids) | set(test_ids)):
        raise ValueError("Fit/heldout article overlap")
    if len(set(md5s)) != len(md5s):
        raise ValueError("Duplicate article identifiers")
    positions = {md5: i for i, md5 in enumerate(md5s)}
    # Keep the training identifier order fixed for dictionary and document tags.
    training_ids = json.loads((source / "train_md5_list.json").read_text())
    if set(training_ids) != set(
        json.loads((evidence / "lda_fit_article_md5s.json").read_text())
    ):
        raise ValueError("Released fit IDs disagree with split manifest")
    train_positions = np.array([positions[md5] for md5 in training_ids])
    dump(output / "config.json", config)
    dump(output / "fit_article_md5s.json", training_ids)
    dump(output / "valid_article_md5s.json", valid_ids)
    dump(output / "test_article_md5s.json", test_ids)
    token_path = output / "tokens.jsonl"
    token_meta = output / "tokens_manifest.json"
    token_fingerprint = {
        "source_sha256": sha256(source / "docs_title_paragraph.json"),
        "tokenizer": config["tokenizer"],
        "nltk_version": nltk.__version__,
        "wordnet_sha256": sha256(Path(str(nltk.data.find("corpora/wordnet.zip")))),
        "punkt_english_sha256": {
            p.name: sha256(p)
            for p in Path(str(nltk.data.find("tokenizers/punkt_tab/english"))).iterdir()
            if p.is_file()
        },
        "stopwords_sha256": hashlib.sha256(
            "\n".join(stopwords.words("english")).encode()
        ).hexdigest(),
    }
    previous = json.loads(token_meta.read_text()) if token_meta.exists() else {}
    if (
        not token_path.exists()
        or previous.get("inputs") != token_fingerprint
        or previous.get("sha256") != sha256(token_path)
    ):
        articles = json.loads((source / "docs_title_paragraph.json").read_text())
        if len(articles) != len(md5s):
            raise ValueError("Article/MD5 count mismatch")
        temporary = token_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for index, tokens in enumerate(clean_documents(articles)):
                stream.write(json.dumps(tokens, ensure_ascii=False) + "\n")
                if (index + 1) % 10000 == 0:
                    print(
                        json.dumps(
                            {
                                "region": region,
                                "tokenized": index + 1,
                                "total": len(md5s),
                            }
                        ),
                        flush=True,
                    )
        os.replace(temporary, token_path)
        dump(token_meta, {"inputs": token_fingerprint, "sha256": sha256(token_path)})
        del articles
    with token_path.open() as stream:
        tokens = [json.loads(line) for line in stream]
    if len(tokens) != len(md5s):
        raise ValueError("Token/article count mismatch")
    train_tokens = [tokens[i] for i in train_positions]
    posteriors = {}
    fit_records = {}
    seed = config["seed"]
    if "lda" in methods:
        settings = config["lda"]
        dictionary = corpora.Dictionary(train_tokens)
        dictionary.save(str(output / "lda_dictionary.gensim"))
        corpus = [dictionary.doc2bow(document) for document in train_tokens]
        model = LdaModel(
            corpus,
            id2word=dictionary,
            num_topics=settings["topics"],
            random_state=seed,
            **{k: v for k, v in settings.items() if k != "topics"},
        )
        model.save(str(output / "lda.gensim"))
        before = arrays_digest(fitted_arrays(model, "lda"))
        posterior = np.zeros((len(tokens), settings["topics"]), dtype=np.float32)
        for i, document in enumerate(tokens):
            # Retain topics above the configured minimum probability.
            for topic, value in model.get_document_topics(dictionary.doc2bow(document)):
                posterior[i, topic] = value
        after = arrays_digest(fitted_arrays(model, "lda"))
        if before != after:
            raise RuntimeError("Heldout inference altered LDA fitted arrays")
        posteriors["lda"] = posterior
        fit_records["lda"] = {
            "fit_parameter_sha256": before,
            "after_assignment_parameter_sha256": after,
            "heldout_updates": False,
            "vocabulary_fit_articles": "fit_article_md5s.json",
        }
        del corpus, model, dictionary
    if set(methods) & {"doc2vec", "kmeans", "gmm"}:
        settings = config["doc2vec"]
        model = Doc2Vec(
            vector_size=settings["vector_size"],
            seed=seed,
            **{
                k: v
                for k, v in settings.items()
                if k not in ["vector_size", "infer_epochs"]
            },
        )
        tagged = [
            TaggedDocument(words, [md5])
            for words, md5 in zip(train_tokens, training_ids)
        ]
        model.build_vocab(tagged)
        model.train(tagged, total_examples=len(tagged), epochs=settings["epochs"])
        model.save(str(output / "doc2vec.gensim"))
        before = arrays_digest(fitted_arrays(model, "doc2vec"))
        vectors = np.empty((len(md5s), settings["vector_size"]), dtype=np.float32)
        train_set = set(training_ids)
        for i, (md5, document) in enumerate(zip(md5s, tokens)):
            if md5 in train_set:
                vectors[i] = model.dv[md5]
            else:
                # A document-specific seed makes assignments independent of heldout order.
                model.random.seed(
                    int(hashlib.sha256((str(seed) + md5).encode()).hexdigest()[:8], 16)
                )
                vectors[i] = model.infer_vector(
                    document, epochs=settings["infer_epochs"]
                )
            if (i + 1) % 10000 == 0:
                print(
                    json.dumps(
                        {
                            "region": region,
                            "doc2vec_assigned": i + 1,
                            "total": len(md5s),
                        }
                    ),
                    flush=True,
                )
        after = arrays_digest(fitted_arrays(model, "doc2vec"))
        if before != after:
            raise RuntimeError("Heldout inference altered doc2vec fitted arrays")
        np.save(output / "doc2vec_vectors.npy", vectors)
        fit_records["doc2vec"] = {
            "fit_parameter_sha256": before,
            "after_assignment_parameter_sha256": after,
            "heldout_updates": False,
            "heldout_method": "infer a new document vector; freeze vocabulary, word and training document vectors",
        }
        del model, tagged
        for method in ["kmeans", "gmm"]:
            if method not in methods:
                continue
            settings = config[method]
            cluster = (
                KMeans(
                    n_clusters=settings["clusters"],
                    random_state=seed,
                    **{k: v for k, v in settings.items() if k != "clusters"},
                )
                if method == "kmeans"
                else GaussianMixture(
                    n_components=settings["components"],
                    random_state=seed,
                    **{k: v for k, v in settings.items() if k != "components"},
                )
            )
            cluster.fit(vectors[train_positions])
            with (output / f"{method}.pkl").open("wb") as stream:
                pickle.dump(cluster, stream)
            before = arrays_digest(fitted_arrays(cluster, method))
            if method == "kmeans":
                posterior = np.eye(settings["clusters"], dtype=np.float32)[
                    cluster.predict(vectors)
                ]
            else:
                posterior = cluster.predict_proba(vectors).astype(np.float32)
            after = arrays_digest(fitted_arrays(cluster, method))
            if before != after:
                raise RuntimeError(f"Heldout assignment altered {method}")
            posteriors[method] = posterior
            fit_records[method] = {
                "fit_parameter_sha256": before,
                "after_assignment_parameter_sha256": after,
                "heldout_updates": False,
                "input_vectors": "doc2vec_vectors.npy",
                "converged": bool(getattr(cluster, "converged_", True)),
                "iterations": int(cluster.n_iter_),
            }
    for method, posterior in posteriors.items():
        np.save(output / f"{method}_article_posterior.npy", posterior)
        np.save(output / f"{method}_article_context.npy", posterior.argmax(axis=1))
    manifest = json.loads((evidence / "manifest.json").read_text())
    # Use the released timestamp ranges for event-to-split assignment.
    boundaries = {
        name: set(
            range(
                manifest["splits"][name]["t_min"], manifest["splits"][name]["t_max"] + 1
            )
        )
        for name in ["train", "valid", "test"]
    }
    from contextlib import ExitStack

    with ExitStack() as stack:
        source_rows = {
            split: stack.enter_context((source / f"{split}.txt").open())
            for split in boundaries
        }
        streams = {
            (method, split): stack.enter_context(
                (output / f"{method}_{split}_w_contextid.txt").open("w")
            )
            for method in posteriors
            for split in boundaries
        }
        article_assignment = stack.enter_context(
            (output / "article_assignments.csv").open("w")
        )
        writer = csv.writer(article_assignment)
        writer.writerow(["article_md5", "split", "used_for_fitting", *posteriors])
        split_sets = {
            "train": set(training_ids),
            "valid": set(valid_ids),
            "test": set(test_ids),
        }
        for i, md5 in enumerate(md5s):
            assigned_split = next(
                (split for split, ids in split_sets.items() if md5 in ids), "unlinked"
            )
            writer.writerow(
                [
                    md5,
                    assigned_split,
                    md5 in split_sets["train"],
                    *[int(p[i].argmax()) for p in posteriors.values()],
                ]
            )
        source_csv = stack.enter_context((source / f"{region}.csv").open())
        entity_map = dict(
            line.rstrip().rsplit("\t", 1)
            for line in (source / "entity2id.txt").read_text().splitlines()
        )
        relation_map = dict(
            line.rstrip().rsplit("\t", 1)
            for line in (source / "relation2id.txt").read_text().splitlines()
        )
        counts = {split: 0 for split in boundaries}
        for event in csv.DictReader(source_csv, delimiter="\t"):
            t = int(event["timid"])
            split = next(split for split, times in boundaries.items() if t in times)
            indices = [positions[md5] for md5 in event["Md5_list"].split(", ")]
            triple = [
                int(entity_map[event["Actor1Name"]]),
                int(relation_map[event["EventCode"]]),
                int(entity_map[event["Actor2Name"]]),
                t,
            ]
            expected = source_rows[split].readline()
            if not expected or triple != list(map(int, expected.split())):
                raise ValueError(
                    "Article-derived event order differs from the released split"
                )
            for method, p in posteriors.items():
                label = int(p[indices].mean(axis=0).argmax())
                streams[method, split].write(
                    "\t".join(map(str, [*triple, label])) + "\n"
                )
            counts[split] += 1
        if any(stream.readline() for stream in source_rows.values()):
            raise ValueError("Article-derived events omit released split rows")
    for split in counts:
        if counts[split] != manifest["splits"][split]["rows"]:
            raise ValueError("Context-labelled rows differ from released splits")
    record = {
        "region": region,
        "status": "completed",
        "fitting_articles": len(training_ids),
        "validation_articles": len(valid_ids),
        "test_articles": len(test_ids),
        "fitting_ids_sha256": sha256(output / "fit_article_md5s.json"),
        "methods": fit_records,
        "split_rows": counts,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "fit_scope": "train articles only; shared doc2vec for K-means and GMM",
        "assignment": "article posterior or hard K-means label; event mean article posterior then argmax",
        "frozen_heldout_parameters_verified": True,
        "config": config,
        "artifacts_sha256": {
            p.name: sha256(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    }
    dump(output / "manifest.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions", nargs="+", choices=["EG", "IR", "IS"], default=["EG", "IR", "IS"]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lda", "doc2vec", "kmeans", "gmm"],
        default=["lda", "kmeans", "gmm"],
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/contexts.json")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/contexts")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    required_env = {
        "PYTHONHASHSEED": str(config["seed"]),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if any(os.environ.get(key) != value for key, value in required_env.items()):
        os.execve(
            sys.executable,
            [sys.executable, str(ROOT / "scripts/fit_contexts.py"), *sys.argv[1:]],
            {
                **os.environ,
                **required_env,
            },
        )
    for region in args.regions:
        fit_region(
            region,
            config,
            args.methods,
            args.output_root,
            args.data_root,
            args.prepared_root,
        )


if __name__ == "__main__":
    main()
