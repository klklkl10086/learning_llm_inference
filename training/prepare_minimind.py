"""下载固定版本 MiniMind 的中文小子集，并按文档划分训练和验证集。"""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mib", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    repo = "jingyaogong/minimind_dataset"
    filename = "pretrain_t2t_mini.jsonl"
    url = f"https://huggingface.co/datasets/{repo}/resolve/{args.revision}/{filename}"
    target = args.target_mib * 1024 * 1024
    counts = {"train": 0, "val": 0}
    byte_counts = {"train": 0, "val": 0}
    seen = set()
    accepted_bytes = 0
    source_lines = 0
    next_report = 2 * 1024 * 1024

    print(f"Source: {repo}/{filename}, revision={args.revision}", flush=True)
    # 即使服务端忽略 Range，达到目标后也会关闭流，不下载整个数据集。
    headers = {"Range": "bytes=0-134217727", "Accept-Encoding": "identity"}
    with (
        (args.output_dir / "train.jsonl").open("w", encoding="utf-8") as train_file,
        (args.output_dir / "val.jsonl").open("w", encoding="utf-8") as val_file,
        requests.get(url, headers=headers, stream=True, timeout=(20, 60)) as response,
    ):
        response.raise_for_status()
        outputs = {"train": train_file, "val": val_file}
        for line in response.iter_lines(chunk_size=65536):
            if not line:
                continue
            source_lines += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            text = record.get("text")
            if not isinstance(text, str):
                continue
            text = text.replace("\r\n", "\n").strip()
            if not 80 <= len(text) <= 6000:
                continue
            cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
            if cjk / len(text) < 0.25:
                continue

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            split = "val" if int(digest[:8], 16) % 100 < 5 else "train"
            outputs[split].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            size = len(text.encode("utf-8"))
            counts[split] += 1
            byte_counts[split] += size
            accepted_bytes += size
            if accepted_bytes >= next_report:
                print(f"Accepted {accepted_bytes / 1024**2:.1f} MiB; docs={sum(counts.values())}", flush=True)
                next_report += 2 * 1024 * 1024
            if accepted_bytes >= target:
                break

    if accepted_bytes < target or not all(counts.values()):
        raise RuntimeError("没有读到足够语料，请检查网络或放宽子集大小")
    metadata = {
        "repo_id": repo,
        "revision": args.revision,
        "file": filename,
        "url": url,
        "selection": "source-order prefix, text length 80..6000, CJK fraction >= 0.25",
        "split": "SHA256(text) first 8 hex digits mod 100 < 5 -> validation",
        "deduplication": "exact normalized-text SHA256",
        "source_lines_read": source_lines,
        "documents": counts,
        "utf8_text_bytes": byte_counts,
        "total_utf8_text_bytes": accepted_bytes,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
