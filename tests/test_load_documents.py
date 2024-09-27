from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.concept_web.load_documents import (extract_lesson_objectives,
                                            extract_lessons_from_page,
                                            extract_text_from_pdf,
                                            find_docx_indices,
                                            find_pdf_lessons,
                                            infer_lesson_from_filename,
                                            infer_lesson_number,
                                            load_documents, load_docx_syllabus,
                                            load_lessons, load_readings)


@pytest.fixture
def sample_directory(tmp_path):
    # Create a temporary directory with sample files for testing
    sample_dir = tmp_path / "lessons"
    sample_dir.mkdir()

    # Create sample files
    (sample_dir / "Lesson1.txt").write_text("Sample text for lesson 1")
    (sample_dir / "Lesson2.pdf").write_bytes(b"PDF content")
    (sample_dir / "Lesson3.docx").write_text("Sample DOCX content")

    return sample_dir


def test_load_documents(sample_directory):
    lesson_number = 1
    documents = load_documents(sample_directory, lesson_number)
    assert len(documents) == 1
    assert "Sample text for lesson 1" in documents[0]


def test_load_lessons(sample_directory):
    lesson_range = range(1, 3)
    documents = load_lessons(sample_directory, lesson_range=lesson_range)
    assert len(documents) == 2


def test_infer_lesson_number():
    path = Path("Lesson1")
    lesson_number = infer_lesson_number(path, infer_from="filename")
    assert lesson_number == 1


def test_infer_lesson_from_filename():
    filename = "Lesson2.pdf"
    lesson_number = infer_lesson_from_filename(filename)
    assert lesson_number == 2


@patch("PyPDF2.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader):
    mock_pdf_reader.return_value.pages = [mock_open(read_data="Page 1 text")]
    text = extract_text_from_pdf("dummy.pdf")
    assert "Page 1 text" in text


def test_extract_lessons_from_page():
    content = "Week 1: Introduction\nWeek 2: History"
    lessons = extract_lessons_from_page(content)
    assert len(lessons) == 2
    assert "Introduction" in lessons["Lesson 1"]


def test_find_pdf_lessons():
    syllabus = {"Lesson 1": "Introduction", "Lesson 2": "History"}
    prev, curr, next_, end = find_pdf_lessons(syllabus, 1)
    assert curr == "Introduction"
    assert next_ == "History"


def test_find_docx_indices():
    syllabus_content = ["Lesson 1: Introduction", "Lesson 2: History"]
    prev, curr, next_, end = find_docx_indices(syllabus_content, 1)
    assert curr == 0


def test_load_docx_syllabus():
    with patch("docx.Document") as mock_doc:
        mock_doc.return_value.paragraphs = [mock_open(read_data="Introduction")]
        content = load_docx_syllabus("dummy.docx")
        assert "Introduction" in content[0]


def test_extract_lesson_objectives(sample_directory):
    objectives = extract_lesson_objectives(sample_directory / "Lesson1.txt", 1)
    assert "Sample text" in objectives


def test_load_readings(sample_directory):
    reading = load_readings(sample_directory / "Lesson1.txt")
    assert "Sample text for lesson 1" in reading


if __name__ == "__main__":
    # Run all tests when the script is executed
    pytest.main([__file__])
