"""Unit and integration tests for SpamGuard AI Phase 02 dataset preparation."""

import email.message
import tarfile
from collections import defaultdict
from io import BytesIO

import pytest

from src.config import SPAMASSASSIN_ARCHIVES
from src.prepare_data import (
    MAX_MEMBER_SIZE_BYTES,
    build_dataset_dataframe,
    decode_mime_subject,
    deduplicate_and_validate_records,
    extract_safe_text_from_html,
    is_safe_tar_member,
    normalize_text,
    parse_email_bytes,
)


# =====================================================================
# Task 8 Tests: Unit Tests with Synthetic Fixtures
# =====================================================================

def test_label_mapping_configuration():
    """Verify that spam and ham labels conform to project specifications (Ham=0, Spam=1)."""
    ham_groups = {"easy_ham", "easy_ham_2", "hard_ham"}
    spam_groups = {"spam", "spam_2"}

    for item in SPAMASSASSIN_ARCHIVES:
        group = item["group"]
        label = item["label"]
        if group in ham_groups:
            assert label == 0, f"Expected {group} to be mapped to label 0, got {label}"
        elif group in spam_groups:
            assert label == 1, f"Expected {group} to be mapped to label 1, got {label}"
        else:
            pytest.fail(f"Unexpected group {group} in SPAMASSASSIN_ARCHIVES")


def test_plain_text_extraction():
    """Verify plain-text email parsing extracts subject and body cleanly."""
    raw_email = (
        b"From: sender@example.com\r\n"
        b"To: recipient@example.com\r\n"
        b"Subject: Weekly Project Meeting\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"The weekly sprint sync will take place at 10:00 AM in Conference Room B.\r\n"
    )
    subject, body, error = parse_email_bytes(raw_email)
    assert error is None
    assert subject == "Weekly Project Meeting"
    assert "weekly sprint sync will take place" in body


def test_mime_subject_decoding():
    """Verify MIME-encoded subjects across encodings (UTF-8, ISO-8859-1) decode properly."""
    # Plain subject
    assert decode_mime_subject("Simple ASCII Subject") == "Simple ASCII Subject"

    # UTF-8 Base64: "SpamGuard AI Project"
    utf8_b64 = "=?UTF-8?B?U3BhbUd1YXJkIEFJIFByb2plY3Q=?="
    assert decode_mime_subject(utf8_b64) == "SpamGuard AI Project"

    # ISO-8859-1 Quoted-Printable: "Café Invitation"
    iso_qp = "=?ISO-8859-1?Q?Caf=E9_Invitation?="
    assert decode_mime_subject(iso_qp) == "Café Invitation"

    # None and empty
    assert decode_mime_subject(None) == ""
    assert decode_mime_subject("") == ""


def test_multipart_handling_and_attachment_exclusion():
    """Verify multipart email extraction prefers text/plain and excludes attachments."""
    msg = email.message.EmailMessage()
    msg["Subject"] = "Quarterly Financial Overview"
    msg["From"] = "finance@example.com"
    msg["To"] = "team@example.com"

    # Add text/plain part
    msg.set_content("Please find attached the quarterly financial report.")

    # Add HTML alternative part
    msg.add_alternative(
        "<html><body><p>Please find attached the <b>quarterly financial report</b>.</p></body></html>",
        subtype="html",
    )

    # Add binary attachment
    msg.add_attachment(
        b"%PDF-1.4 dummy pdf bytes for test",
        maintype="application",
        subtype="pdf",
        filename="report.pdf",
    )

    raw_bytes = msg.as_bytes()
    subject, body, error = parse_email_bytes(raw_bytes)

    assert error is None
    assert subject == "Quarterly Financial Overview"
    # Should contain plain text
    assert "Please find attached the quarterly financial report." in body
    # Should NOT contain binary attachment bytes
    assert "%PDF" not in body


def test_html_fallback_behavior():
    """Verify that when only HTML is available, text is safely extracted without scripts."""
    raw_email = (
        b"From: promo@offers.com\r\n"
        b"Subject: Claim Your Gift Voucher\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b"<!DOCTYPE html>\r\n"
        b"<html>\r\n"
        b"<head>\r\n"
        b"<title>Ignored Title</title>\r\n"
        b"<script>alert('malicious script execution attempt');</script>\r\n"
        b"<style>body { color: red; }</style>\r\n"
        b"</head>\r\n"
        b"<body>\r\n"
        b"<h1>Exclusive Member Deal!</h1>\r\n"
        b"<p>Click <a href='https://example.com/claim'>here</a> to claim your $250 reward.</p>\r\n"
        b"</body>\r\n"
        b"</html>\r\n"
    )
    subject, body, error = parse_email_bytes(raw_email)

    assert error is None
    assert subject == "Claim Your Gift Voucher"
    assert "Exclusive Member Deal!" in body
    assert "$250 reward" in body
    # Disallowed tags content must not be present
    assert "malicious script" not in body
    assert "color: red" not in body


def test_empty_message_rejection():
    """Verify that empty emails or whitespace-only messages are identified and rejected."""
    raw_empty = b"From: ghost@example.com\r\nSubject:   \r\n\r\n    \r\n"
    subject, body, error = parse_email_bytes(raw_empty)
    assert error is None
    combined = f"{subject}\n\n{body}"
    normalized = normalize_text(combined)
    assert normalized == ""


def test_duplicate_detection():
    """Verify duplicate messages with identical normalized content keep only one instance."""
    records = [
        {
            "message_id": "archive_1:0001",
            "subject": "Discount Pharmacy",
            "body": "Buy cheap medications online at discounts up to 70%.",
            "text": "Discount Pharmacy\n\nBuy cheap medications online at discounts up to 70%.",
            "label": 1,
            "source_group": "spam",
            "source_archive": "archive_1.tar.bz2",
            "content_hash": "hash_pharma_123",
        },
        {
            "message_id": "archive_2:0099",
            "subject": "Discount Pharmacy",
            "body": "Buy cheap medications online at discounts up to 70%.",
            "text": "Discount Pharmacy\n\nBuy cheap medications online at discounts up to 70%.",
            "label": 1,
            "source_group": "spam_2",
            "source_archive": "archive_2.tar.bz2",
            "content_hash": "hash_pharma_123",
        },
    ]

    stats = defaultdict(int)
    stats["conflicting_details"] = []
    usable = deduplicate_and_validate_records(records, stats)

    assert len(usable) == 1
    assert stats["removed_duplicates"] == 1
    assert usable[0]["message_id"] == "archive_1:0001"


def test_conflicting_label_handling():
    """Verify conflicting labels for identical text are excluded completely from output."""
    records = [
        {
            "message_id": "ham_archive:001",
            "subject": "Notice of Update",
            "body": "Please review the attached terms update.",
            "text": "Notice of Update\n\nPlease review the attached terms update.",
            "label": 0,  # Labeled Ham
            "source_group": "easy_ham",
            "source_archive": "easy_ham.tar.bz2",
            "content_hash": "conflict_hash_999",
        },
        {
            "message_id": "spam_archive:002",
            "subject": "Notice of Update",
            "body": "Please review the attached terms update.",
            "text": "Notice of Update\n\nPlease review the attached terms update.",
            "label": 1,  # Labeled Spam
            "source_group": "spam",
            "source_archive": "spam.tar.bz2",
            "content_hash": "conflict_hash_999",
        },
    ]

    stats = defaultdict(int)
    stats["conflicting_details"] = []
    usable = deduplicate_and_validate_records(records, stats)

    # Both must be rejected due to label conflict
    assert len(usable) == 0
    assert stats["conflicting_label_records"] == 2
    assert len(stats["conflicting_details"]) == 1


def test_unexpected_archive_entries():
    """Verify security rejection of non-files, path traversals, oversized members, and 'cmds'."""
    # 1. Symlink entry
    symlink_member = tarfile.TarInfo("easy_ham/symlink_target")
    symlink_member.type = tarfile.SYMTYPE
    is_safe, reason = is_safe_tar_member(symlink_member)
    assert not is_safe
    assert "Non-regular file" in reason

    # 2. Directory entry
    dir_member = tarfile.TarInfo("easy_ham")
    dir_member.type = tarfile.DIRTYPE
    is_safe, reason = is_safe_tar_member(dir_member)
    assert not is_safe

    # 3. Path traversal entry
    traversal_member = tarfile.TarInfo("easy_ham/../../etc/passwd")
    traversal_member.type = tarfile.REGTYPE
    is_safe, reason = is_safe_tar_member(traversal_member)
    assert not is_safe
    assert "traversal" in reason.lower()

    # 4. Oversized entry
    oversized_member = tarfile.TarInfo("easy_ham/huge_email.eml")
    oversized_member.type = tarfile.REGTYPE
    oversized_member.size = MAX_MEMBER_SIZE_BYTES + 1024
    is_safe, reason = is_safe_tar_member(oversized_member)
    assert not is_safe
    assert "exceeds" in reason.lower()

    # 5. Corpus 'cmds' script
    cmds_member = tarfile.TarInfo("easy_ham/cmds")
    cmds_member.type = tarfile.REGTYPE
    cmds_member.size = 2048
    is_safe, reason = is_safe_tar_member(cmds_member)
    assert not is_safe
    assert "cmds" in reason.lower()

    # 6. Valid regular email member
    valid_member = tarfile.TarInfo("easy_ham/0001.20020822")
    valid_member.type = tarfile.REGTYPE
    valid_member.size = 4096
    is_safe, reason = is_safe_tar_member(valid_member)
    assert is_safe
    assert reason == ""


def test_valid_output_schema():
    """Verify build_dataset_dataframe enforces expected columns, non-null values, and binary labels."""
    valid_records = [
        {
            "message_id": "arch1:001",
            "subject": "Hello",
            "body": "Good morning team",
            "text": "Hello\n\nGood morning team",
            "label": 0,
            "source_group": "easy_ham",
            "source_archive": "arch1.tar.bz2",
        },
        {
            "message_id": "arch2:002",
            "subject": "Win Cash",
            "body": "Click here to claim $1000",
            "text": "Win Cash\n\nClick here to claim $1000",
            "label": 1,
            "source_group": "spam",
            "source_archive": "arch2.tar.bz2",
        },
    ]

    df = build_dataset_dataframe(valid_records)
    expected_cols = [
        "message_id",
        "subject",
        "body",
        "text",
        "label",
        "source_group",
        "source_archive",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
    assert set(df["label"].unique()) == {0, 1}

    # Test rejection of missing spam class
    ham_only = [r for r in valid_records if r["label"] == 0]
    with pytest.raises(ValueError, match="must contain both ham.*and spam"):
        build_dataset_dataframe(ham_only)

    # Test rejection of invalid label value
    invalid_label_record = [
        {
            "message_id": "arch1:003",
            "subject": "Test",
            "body": "Body",
            "text": "Test\n\nBody",
            "label": 99,
            "source_group": "unknown",
            "source_archive": "arch.tar.bz2",
        }
    ]
    with pytest.raises(ValueError, match="Invalid labels found"):
        build_dataset_dataframe(invalid_label_record)


def test_deterministic_data_preparation():
    """Verify deduplication yields identical output rows and ordering regardless of input sequence."""
    r1 = {
        "message_id": "b_archive:001",
        "subject": "Subject B",
        "body": "Content B",
        "text": "Subject B\n\nContent B",
        "label": 0,
        "source_group": "easy_ham",
        "source_archive": "b.tar.bz2",
        "content_hash": "hash_b",
    }
    r2 = {
        "message_id": "a_archive:002",
        "subject": "Subject A",
        "body": "Content A",
        "text": "Subject A\n\nContent A",
        "label": 1,
        "source_group": "spam",
        "source_archive": "a.tar.bz2",
        "content_hash": "hash_a",
    }
    r3 = {
        "message_id": "c_archive:003",
        "subject": "Subject A",
        "body": "Content A",
        "text": "Subject A\n\nContent A",
        "label": 1,
        "source_group": "spam_2",
        "source_archive": "c.tar.bz2",
        "content_hash": "hash_a",
    }

    stats1 = defaultdict(int)
    stats1["conflicting_details"] = []
    run1 = deduplicate_and_validate_records([r1, r2, r3], stats1)

    stats2 = defaultdict(int)
    stats2["conflicting_details"] = []
    run2 = deduplicate_and_validate_records([r3, r1, r2], stats2)

    assert len(run1) == 2
    assert len(run2) == 2
    # Verify deterministic ordering by message_id
    assert [x["message_id"] for x in run1] == [x["message_id"] for x in run2]
    assert [x["message_id"] for x in run1] == ["a_archive:002", "b_archive:001"]
