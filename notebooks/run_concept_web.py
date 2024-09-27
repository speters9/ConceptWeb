import os
from pathlib import Path

# env setup
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from src.concept_web.ConceptWeb import ConceptMapBuilder

load_dotenv()

# llm chain setup
# Path definitions

# readings saved here (could be directory of readings, or directory of subdirectories, where each subdirectory represents a lesson)
readingDir = Path(os.getenv('readingsDir'))

# path to syllabus (.pdf or .docx)
syllabus_path = Path(os.getenv('syllabus_path'))
pdf_syllabus_path = Path(os.getenv('pdf_syllabus_path'))

# root directory
projectDir = here()

# %%
# Example usage
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv('openai_key'),
    organization=os.getenv('openai_org'),
)


builder = ConceptMapBuilder(
    readings_dir=readingDir,
    project_dir=projectDir,
    syllabus_path=syllabus_path,
    llm=llm,
    course_name="American Politics",
    lesson_range=range(11, 16),
    recursive=True,
    verbose=True
)

builder.run_full_pipeline()
