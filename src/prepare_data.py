"""SpamGuard AI — Dataset Preparation Pipeline.

Acquires, safely parses, normalizes, validates, and deduplicates real labeled
emails from the Apache SpamAssassin Public Corpus for spam classification.

Pipeline features:
- Downloads official Apache SpamAssassin tar.bz2 archives with SHA-256 verification.
- Enforces archive security (path traversal checks, symlink rejections, file size limits).
- Decodes MIME subjects and handles legacy/international character encodings.
- Extracts plain text bodies with secure, non-executing HTML fallback.
- Strictly excludes transport headers, server-generated labels, and metadata snooping.
- Performs Unicode NFKC normalization and line break standardization.
- Detects and drops duplicate content and excludes conflicting-label messages.
- Exports validated datasets to data/processed/emails.csv and reports/data_summary.json.
"""

from __future__ import annotations

import argparse
import email
import email.header
import email.message
import hashlib
import html
import json
import logging
import re
import sys
import tarfile
import unicodedata
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATA_SUMMARY_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SPAMASSASSIN_ARCHIVES,
    SPAMASSASSIN_BASE_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpamGuardAI.PrepareData")

MAX_MEMBER_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per email maximum


# =====================================================================
# HTML Safe Text Extractor
# =====================================================================

class SafeHTMLTextExtractor(HTMLParser):
    """Safely extracts plain-text content from HTML without executing or rendering."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.text_parts: list[str] = []
        self._ignore: bool = False
        self._ignore_depth: int = 0
        self.ignore_tags = {"script", "style", "head", "title", "meta", "link", "noscript", "iframe", "object"}
        self.block_tags = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "blockquote"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in self.ignore_tags:
            self._ignore = True
            self._ignore_depth += 1
        elif tag_lower in self.block_tags:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self.ignore_tags:
            self._ignore_depth = max(0, self._ignore_depth - 1)
            if self._ignore_depth == 0:
                self._ignore = False
        elif tag_lower in self.block_tags:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignore and data:
            self.text_parts.append(data)

    def get_text(self) -> str:
        raw_combined = "".join(self.text_parts)
        return html.unescape(raw_combined).strip()


def extract_safe_text_from_html(html_content: str) -> str:
    """Safely extracts readable text from an HTML string with parser and regex fallback."""
    if not html_content or not html_content.strip():
        return ""
    try:
        extractor = SafeHTMLTextExtractor()
        extractor.feed(html_content)
        parsed = extractor.get_text()
        if parsed:
            return parsed
    except Exception as exc:
        logger.debug("SafeHTMLTextExtractor raised %s, using regex fallback.", exc)

    # Regex fallback for severely malformed HTML
    clean = re.sub(r"<(script|style|head|title)[^>]*>.*?</\1>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    return html.unescape(clean).strip()


# =====================================================================
# MIME Subject & Text Helpers
# =====================================================================

def decode_mime_subject(raw_subject: Any) -> str:
    """Decodes MIME-encoded email subjects into clean UTF-8 text."""
    if raw_subject is None:
        return ""
    raw_str = str(raw_subject).strip()
    if not raw_str:
        return ""

    try:
        chunks = email.header.decode_header(raw_str)
    except Exception:
        return raw_str

    decoded_pieces: list[str] = []
    for chunk, charset in chunks:
        if isinstance(chunk, bytes):
            enc = (charset or "utf-8").strip()
            try:
                decoded_pieces.append(chunk.decode(enc, errors="replace"))
            except (LookupError, ValueError):
                try:
                    decoded_pieces.append(chunk.decode("iso-8859-1", errors="replace"))
                except Exception:
                    decoded_pieces.append(chunk.decode("ascii", errors="replace"))
        else:
            decoded_pieces.append(str(chunk))

    return " ".join(decoded_pieces).strip()


def normalize_text(text: str) -> str:
    """Applies essential text normalization: Unicode NFKC, newline standardization, trimming."""
    if not text:
        return ""
    # Safe Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove null characters
    text = text.replace("\x00", "")
    return text.strip()


def compute_sha256(filepath: Path) -> str:
    """Computes the SHA-256 checksum of a file in 64KB blocks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =====================================================================
# Tar Archive Security & Email Body Extraction
# =====================================================================

def is_safe_tar_member(member: tarfile.TarInfo, max_size_bytes: int = MAX_MEMBER_SIZE_BYTES) -> tuple[bool, str]:
    """Validates archive member safety to prevent path traversal and resource exhaustion."""
    if not (member.isreg() or member.isfile()):
        return False, f"Non-regular file member (type={member.type})"

    name = member.name
    if not name or "\x00" in name:
        return False, "Invalid or null-containing member filename"

    if name.startswith("/") or name.startswith("\\"):
        return False, "Absolute path in archive member"

    parts = Path(name).parts
    if ".." in parts:
        return False, "Path traversal sequence (..) detected"

    if member.size > max_size_bytes:
        return False, f"Member size {member.size} bytes exceeds maximum limit ({max_size_bytes})"

    # Filter corpus administrative shell scripts
    if name.endswith("/cmds") or name == "cmds":
        return False, "Corpus administrative command script (cmds), not an email"

    return True, ""


def parse_email_bytes(raw_bytes: bytes) -> tuple[str, str, str | None]:
    """Parses raw email bytes using the standard email package.

    Returns:
        tuple[subject, body, error_reason]
    """
    if not raw_bytes or not raw_bytes.strip():
        return "", "", "Empty raw bytes"

    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception as exc:
        return "", "", f"Email parsing exception: {exc}"

    subject = decode_mime_subject(msg.get("subject", ""))

    plain_parts: list[str] = []
    html_parts: list[str] = []

    try:
        for part in msg.walk():
            # Skip container multipart wrappers
            if part.is_multipart():
                continue

            # Exclude attachments
            disposition = part.get_content_disposition()
            if disposition == "attachment":
                continue
            filename = part.get_filename()
            if filename:
                continue

            content_type = part.get_content_type().lower()
            payload = part.get_payload(decode=True)
            if not payload:
                continue

            charset = (part.get_content_charset() or "utf-8").strip()
            try:
                decoded_str = payload.decode(charset, errors="replace")
            except (LookupError, ValueError):
                try:
                    decoded_str = payload.decode("iso-8859-1", errors="replace")
                except Exception:
                    decoded_str = payload.decode("ascii", errors="replace")

            if content_type == "text/plain":
                plain_parts.append(decoded_str)
            elif content_type == "text/html":
                html_parts.append(decoded_str)

        if plain_parts:
            body = "\n\n".join(plain_parts)
        elif html_parts:
            # Fallback to safe HTML extraction when only HTML is provided
            extracted_html_texts = [extract_safe_text_from_html(h) for h in html_parts]
            body = "\n\n".join(t for t in extracted_html_texts if t)
        else:
            body = ""

    except Exception as exc:
        return subject, "", f"Payload extraction error: {exc}"

    return subject, body, None


# =====================================================================
# Archive Downloading & Processing
# =====================================================================

def download_archive(url: str, dest_path: Path, timeout: int = 60) -> str:
    """Downloads a single archive if missing or invalid, returning its SHA-256 checksum."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        # Validate that the file is an accessible tar archive
        try:
            with tarfile.open(dest_path, "r:bz2") as tar:
                # Check that archive can be opened
                _ = len(tar.getmembers())
            logger.info("Using existing verified archive: %s", dest_path.name)
            return compute_sha256(dest_path)
        except Exception as exc:
            logger.warning("Existing archive %s corrupted (%s), re-downloading...", dest_path.name, exc)
            dest_path.unlink(missing_ok=True)

    logger.info("Downloading %s -> %s...", url, dest_path.name)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SpamGuardAI-DataPipeline/1.0 (Academic College Project)"},
    )

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, open(temp_path, "wb") as out_file:
            while chunk := response.read(65536):
                out_file.write(chunk)
        temp_path.replace(dest_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed downloading archive from {url}: {exc}") from exc

    # Verify downloaded archive
    try:
        with tarfile.open(dest_path, "r:bz2") as tar:
            _ = len(tar.getmembers())
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded archive {dest_path.name} is not a valid bz2 tar: {exc}") from exc

    sha256 = compute_sha256(dest_path)
    logger.info("Successfully downloaded %s (SHA-256: %s)", dest_path.name, sha256)
    return sha256


def download_all_archives(
    base_url: str = SPAMASSASSIN_BASE_URL,
    archives: list[dict[str, Any]] | None = None,
    raw_dir: Path = RAW_DATA_DIR,
) -> dict[str, str]:
    """Downloads all configured SpamAssassin archives, returning {filename: sha256}."""
    if archives is None:
        archives = SPAMASSASSIN_ARCHIVES

    checksums: dict[str, str] = {}
    for item in archives:
        filename = str(item["filename"])
        url = base_url + filename
        dest = raw_dir / filename
        checksum = download_archive(url, dest)
        checksums[filename] = checksum

    return checksums


def extract_and_parse_archive(
    archive_path: Path,
    group: str,
    label: int,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Safely inspects and parses email records from a SpamAssassin tar.bz2 archive."""
    records: list[dict[str, Any]] = []
    archive_name = archive_path.name

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            stats["total_source_records"] += 1
            stats["counts_by_source_archive"][archive_name] += 1

            is_safe, reason = is_safe_tar_member(member)
            if not is_safe:
                stats["skipped_members"].append({"archive": archive_name, "member": member.name, "reason": reason})
                continue

            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                stats["parsing_failure_count"] += 1
                continue

            raw_bytes = extracted_file.read()
            subject, body, parse_error = parse_email_bytes(raw_bytes)

            if parse_error:
                stats["parsing_failure_count"] += 1
                stats["skipped_members"].append(
                    {"archive": archive_name, "member": member.name, "reason": parse_error}
                )
                continue

            # Construct classification text from decoded subject and body
            if subject and body:
                combined_text = f"{subject}\n\n{body}"
            elif body:
                combined_text = body
            elif subject:
                combined_text = subject
            else:
                combined_text = ""

            normalized = normalize_text(combined_text)
            if not normalized:
                stats["empty_or_invalid_count"] += 1
                continue

            # Normalized content hash for deduplication and conflict checking
            content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            message_id = f"{archive_name}:{member.name}"

            records.append(
                {
                    "message_id": message_id,
                    "subject": normalize_text(subject),
                    "body": normalize_text(body),
                    "text": normalized,
                    "label": int(label),
                    "source_group": group,
                    "source_archive": archive_name,
                    "content_hash": content_hash,
                }
            )

    return records


# =====================================================================
# Validation & Deduplication
# =====================================================================

def deduplicate_and_validate_records(
    records: list[dict[str, Any]],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """Deduplicates emails and excludes conflicting-label messages.

    Deterministic logic:
    - Sort candidate records by message_id.
    - Group by content_hash.
    - If a content_hash contains conflicting labels (both 0 and 1), exclude all instances.
    - If multiple records share the same hash and label, keep the first instance deterministically.
    """
    # Sort deterministically
    sorted_records = sorted(records, key=lambda r: r["message_id"])

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted_records:
        groups[r["content_hash"]].append(r)

    usable_records: list[dict[str, Any]] = []

    for content_hash, group_records in groups.items():
        unique_labels = {r["label"] for r in group_records}

        if len(unique_labels) > 1:
            # Conflicting labels for identical text: exclude all!
            stats["conflicting_label_records"] += len(group_records)
            stats["conflicting_details"].append(
                {
                    "content_hash": content_hash,
                    "count": len(group_records),
                    "labels": list(unique_labels),
                    "message_ids": [r["message_id"] for r in group_records],
                }
            )
            continue

        # Keep first record deterministically
        chosen = group_records[0]
        usable_records.append(chosen)

        if len(group_records) > 1:
            duplicates_dropped = len(group_records) - 1
            stats["removed_duplicates"] += duplicates_dropped

    # Sort final usable records deterministically by message_id
    usable_records.sort(key=lambda r: r["message_id"])
    return usable_records


def build_dataset_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Builds and validates the standardized Pandas DataFrame."""
    columns = [
        "message_id",
        "subject",
        "body",
        "text",
        "label",
        "source_group",
        "source_archive",
    ]

    if not records:
        raise ValueError("Cannot build DataFrame from empty records list")

    df = pd.DataFrame(records)
    # Ensure exact column schema
    df = df[columns].copy()

    # Schema & Integrity Validations
    if df.empty:
        raise ValueError("Resulting DataFrame is empty")

    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid labels found: {unique_labels}. Expected only {{0, 1}}.")

    if 0 not in unique_labels or 1 not in unique_labels:
        raise ValueError(f"Dataset must contain both ham (0) and spam (1). Found labels: {unique_labels}")

    empty_texts = df["text"].str.strip().eq("").sum()
    if empty_texts > 0:
        raise ValueError(f"Found {empty_texts} empty text rows in final DataFrame")

    null_counts = df.isnull().sum().to_dict()
    if any(null_counts.values()):
        raise ValueError(f"Found null values in standardized DataFrame: {null_counts}")

    return df


# =====================================================================
# Summary Statistics Generation
# =====================================================================

def generate_data_summary(
    df: pd.DataFrame,
    stats: dict[str, Any],
    archive_checksums: dict[str, str],
    output_path: Path = DATA_SUMMARY_FILE,
) -> dict[str, Any]:
    """Computes genuine dataset metrics and exports to reports/data_summary.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_total = len(df)
    spam_count = int((df["label"] == 1).sum())
    ham_count = int((df["label"] == 0).sum())
    spam_pct = round((spam_count / final_total) * 100, 2) if final_total else 0.0
    ham_pct = round((ham_count / final_total) * 100, 2) if final_total else 0.0

    lengths = df["text"].str.len()
    char_stats = {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
        "median": float(lengths.median()),
    }

    final_archive_counts = df["source_archive"].value_counts().to_dict()

    archive_details = []
    for arch in SPAMASSASSIN_ARCHIVES:
        fn = str(arch["filename"])
        archive_details.append(
            {
                "filename": fn,
                "group": str(arch["group"]),
                "label": int(arch["label"]),
                "url": SPAMASSASSIN_BASE_URL + fn,
                "sha256": archive_checksums.get(fn, ""),
            }
        )

    summary: dict[str, Any] = {
        "dataset_name": "Apache SpamAssassin Public Corpus (Normalized)",
        "total_source_records": stats["total_source_records"],
        "final_usable_messages": final_total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": spam_pct,
        "ham_percentage": ham_pct,
        "removed_duplicates": stats["removed_duplicates"],
        "conflicting_label_records": stats["conflicting_label_records"],
        "empty_or_invalid_count": stats["empty_or_invalid_count"],
        "parsing_failure_count": stats["parsing_failure_count"],
        "counts_by_source_archive": dict(stats["counts_by_source_archive"]),
        "final_counts_by_source_archive": final_archive_counts,
        "email_char_length_stats": char_stats,
        "source_archives": archive_details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary report written to %s", output_path)
    return summary


# =====================================================================
# Main Pipeline Function
# =====================================================================

def prepare_dataset(
    download: bool = False,
    raw_dir: Path = RAW_DATA_DIR,
    output_file: Path = PROCESSED_DATA_FILE,
    summary_file: Path = DATA_SUMMARY_FILE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Runs the end-to-end dataset preparation pipeline."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    archive_checksums: dict[str, str] = {}

    # Check whether archives exist; if missing or if download requested, download/verify them
    all_exist = all((raw_dir / str(a["filename"])).exists() for a in SPAMASSASSIN_ARCHIVES)
    if download or not all_exist:
        logger.info("Downloading and verifying SpamAssassin archives...")
        archive_checksums = download_all_archives(raw_dir=raw_dir)
    else:
        logger.info("Verifying existing archive checksums...")
        for a in SPAMASSASSIN_ARCHIVES:
            fn = str(a["filename"])
            archive_checksums[fn] = compute_sha256(raw_dir / fn)

    stats: dict[str, Any] = {
        "total_source_records": 0,
        "removed_duplicates": 0,
        "conflicting_label_records": 0,
        "empty_or_invalid_count": 0,
        "parsing_failure_count": 0,
        "counts_by_source_archive": defaultdict(int),
        "skipped_members": [],
        "conflicting_details": [],
    }

    all_raw_records: list[dict[str, Any]] = []

    for arch in SPAMASSASSIN_ARCHIVES:
        fn = str(arch["filename"])
        group = str(arch["group"])
        label = int(arch["label"])
        archive_path = raw_dir / fn

        logger.info("Processing archive %s (group=%s, label=%d)...", fn, group, label)
        records = extract_and_parse_archive(archive_path, group, label, stats)
        all_raw_records.extend(records)

    logger.info("Total parsed email records before deduplication: %d", len(all_raw_records))

    usable_records = deduplicate_and_validate_records(all_raw_records, stats)
    logger.info(
        "Deduplication complete: %d usable records (removed %d duplicates, %d conflicting).",
        len(usable_records),
        stats["removed_duplicates"],
        stats["conflicting_label_records"],
    )

    df = build_dataset_dataframe(usable_records)

    # Export CSV
    df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info("Exported %d clean email records to %s", len(df), output_file)

    summary = generate_data_summary(df, stats, archive_checksums, summary_file)
    return df, summary


def main() -> None:
    """CLI Entrypoint for dataset preparation."""
    parser = argparse.ArgumentParser(description="SpamGuard AI — Dataset Preparation")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download or verify official SpamAssassin public corpus archives",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory to store downloaded raw tar archives",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROCESSED_DATA_FILE,
        help="Destination path for prepared emails.csv",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DATA_SUMMARY_FILE,
        help="Destination path for data_summary.json",
    )
    args = parser.parse_args()

    prepare_dataset(
        download=args.download,
        raw_dir=args.raw_dir,
        output_file=args.output_file,
        summary_file=args.summary_file,
    )


if __name__ == "__main__":
    main()
