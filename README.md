# ConceptWeb

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Concept-mapping package for use in social sciences instruction.

## Overview

The Concept Map Builder is a Python-based tool designed to generate concept maps from lesson readings and objectives using a language model (LLM). This tool provides an end-to-end workflow that includes loading documents and lesson objectives, summarizing text, extracting relationships, building graphs, detecting communities, and creating both interactive visualizations and word clouds. It is particularly useful for educators and researchers looking to visually represent the relationships between key concepts in course materials.

[Click here to view an example interactive concept map](reports/ConceptWebOutput/interactive_concept_map.html)

## Features

- **Document Loading:** Supports `.docx`, `.pdf`, and `.txt` file formats for reading lesson materials.
- **Lesson Objective Extraction:** Automatically extracts objectives from provided syllabus documents or accepts user-provided objectives.
    - **Note:** Document loading and lesson objective extraction require clear naming conventions (e.g., directories organized by lesson for document upload, or document naming that suggests its lesson association, like 'Lesson 2 ReadingName'). Within syllabi, lessons associated with the identifier "Lesson" or "Week" will be extracted.
- **Text Summarization:** Uses a language model to summarize lesson readings.
    - **Note:** Example code relies on `langchain_openai.ChatOpenAI` but non-API options such as Ollama should work as well (`langchain_community.llms.Ollama`). Ensure Ollama is up and running first by calling `ollama run llama3` (or whatever version of LLaMA you choose).    
- **Concept Relationship Extraction:** Identifies and processes relationships between key concepts in the summary of the text, using lesson objectives as a guide. Processing steps involve a naive form of entity resolution based on Jaccard similarity.
- **Graph Building:** Constructs an undirected graph from the processed relationships.
- **Community Detection:** Identifies clusters within the graph using algorithms like Leiden, Louvain, or Spectral clustering.
- **Visualization:** Generates an interactive HTML graph and a word cloud to represent the extracted concepts and their relationships.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Sphinx docs
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         concept_web and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── tests              <- Test scripts
│
└── concept_web   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes `concept_web` a Python module.
    │
    ├── ConceptWeb.py           <- The main script orchestrating the entire workflow of generating and visualizing concept maps.
    │
    ├── build_concept_map.py    <- Functions for building the concept map graph, detecting communities, and normalizing node and edge attributes.
    │
    ├── concept_extraction.py   <- Functions for extracting key concepts and relationships from lesson readings using a language model.
    │
    ├── load_documents.py       <- Manages loading and processing of lesson documents (PDF, DOCX, TXT) and extracting lesson objectives from a syllabus.
    │
    ├── prompts.py              <- Contains the predefined prompts used by the language model for summarization and relationship extraction.
    │
    ├── tools.py                <- Utility functions like logger setup and other helpers used throughout the project.
    │
    └── visualize_graph.py      <- Functions for creating interactive graph visualizations and generating word clouds from extracted concepts.
```

--------

## Installation
- **Clone the repository**
  ```
  git clone https://github.com/speters9/ConceptWeb.git
  cd concept_web
  ```
- **Set up environment variables**
  ```
  OPENAI_KEY=your_openai_api_key            # or key for your LLM of choice (not required if using Ollama on your own machine)
  OPENAI_ORG=your_openai_organization_id
  syllabus_path=path/to/your/syllabus.docx  # or
  pdf_syllabus_path=path/to/your/syllabus.pdf
  readingsDir=path/to/your/readings/        # this may be a directory of directories, with each folder containing readings from a lesson; or it may have all readings contained within it
  projectDir=path/to/your/project/
  ```
- **Run the pipeline**
  ```
  from src.concept_web.concept_map_builder import ConceptMapBuilder
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(
      model="gpt-4o-mini",
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      api_key=os.getenv('OPENAI_KEY'),
      organization=os.getenv('OPENAI_ORG'),
  )
  
  ## if using ollama ##
  from langchain_community.llms import Ollama
  llm = Ollama(model="llama3.1",
             temperature=0
             )

  builder = ConceptMapBuilder(
      readings_dir=Path(os.getenv('readingsDir')),
      project_dir=Path(os.getenv('projectDir')),
      syllabus_path=Path(os.getenv('syllabus_path')),
      llm=llm,
      course_name="American Politics",
      lesson_range=range(1, 3),
      recursive=True,
      verbose=True
  )

  builder.run_full_pipeline()
  ```

---

## Usage
### Running the Pipeline (see ```notebooks/``` for example)
  1. Ensure that your lesson readings are stored in the specified readingsDir with the appropriate file extensions (.pdf, .docx, .txt).
  2. Set the lesson range and course name according to your syllabus.
  3. Run the pipeline to generate an interactive HTML graph and word cloud of the extracted concepts.
  4. Concept graph will save in a the root folder under ```reports/```

## Acknowledgements
  - The project uses the LangChain library for language model integration.
  - Special thanks to the open-source community.




