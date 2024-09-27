import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from src.concept_web.build_concept_map import build_graph, detect_communities
from src.concept_web.concept_extraction import (
    extract_concepts_from_relationships, extract_relationships,
    process_relationships, summarize_text)
# Import your existing functions
from src.concept_web.load_documents import (extract_lesson_objectives,
                                            load_lessons)
from src.concept_web.prompts import relationship_prompt, summary_prompt
from src.concept_web.tools import logger_setup
from src.concept_web.visualize_graph import (generate_wordcloud,
                                             visualize_graph_interactive)

# %%


class ConceptMapBuilder:
    """
    A class to generate concept maps from lesson readings and objectives using a language model.

    This class provides an end-to-end workflow to extract, process, and visualize concepts and their relationships
    from lesson materials. It supports loading documents and lesson objectives, summarizing text, extracting relationships,
    building graphs, detecting communities, and creating both interactive visualizations and word clouds.

    It assumes some continuity in lesson directory naming and lesson naming within a given syllabus. Syllabus objectives
    are used to help provide the llm with context for relationship generation.

    Attributes:
        lesson_range (range): The range of lessons to process.
        syllabus_path (Path): Path to the syllabus file containing lesson objectives. Filetypes supported are .docx and .pdf
        readings_dir (Path): Path to the directory containing lesson readings. Filetypes supported are .docx, .txt, and .pdf
        llm (Any): The language model used for text summarization and relationship extraction.
        course_name (str): Name of the course for contextualizing LLM responses.
        recursive (bool): Whether to search for lesson readings recursively in subdirectories.
        concept_list (List[str]): List of unique concepts extracted from the relationships.
        relationship_list (List[Tuple[str, str, str]]): List of relationships between concepts.
        G (nx.Graph): The generated graph of concepts and relationships.
        lesson_objectives (Union[List[str], Dict[str, str]]): User-provided lesson objectives, if any.

    Methods:
        load_lesson_materials():
            Loads the lesson readings and objectives from the specified directories and syllabus.

        summarize_and_extract():
            Summarizes the readings and extracts key relationships using the specified language model.

        build_and_detect_communities():
            Builds a graph from the extracted relationships and detects communities within the graph.

        visualize_graph(output_path: str):
            Generates an interactive graph visualization and saves it as an HTML file.

        generate_wordcloud(output_path: str = None):
            Generates a word cloud of the extracted concepts and optionally saves it as an image file.

        run_pipeline(output_dir: Path):
            Runs the full concept map generation pipeline, from loading materials to saving visualizations.
    """

    def __init__(self, project_dir: Union[str, Path], readings_dir: Union[str, Path], syllabus_path: Union[str, Path],
                 llm, course_name: str, lesson_range: Union[range, int] = None, recursive: bool = False,
                 lesson_objectives: Union[List[str], Dict[str, str]] = None, verbose: bool = True, **kwargs):
        """
        Initializes the ConceptMapBuilder with paths and configurations.

        Args:
            project_dir (Union[str, Path]): The base project directory.
            syllabus_path (Union[str, Path]): The path to the syllabus document (PDF or DOCX).
            llm: The language model instance for summarization and relationship extraction.
            course_name (str): The name of the course (e.g., "American Government").
            lesson_range (range, optional): The range of lesson numbers to load. Defaults to None.
            recursive (bool, optional): Whether to load lessons recursively. Defaults to False.
        """
        self.project_dir = Path(project_dir)
        self.syllabus_path = Path(syllabus_path)
        self.llm = llm
        self.course_name = course_name
        self.lesson_range = range(lesson_range, lesson_range + 1) if isinstance(lesson_range, int) else lesson_range
        self.recursive = recursive
        self.readings_dir = readings_dir
        self.data_dir = self.project_dir / "data"
        self.output_dir = self.project_dir / "reports/ConceptWebOutput"
        self.relationship_list = []
        self.concept_list = []
        self.prompts = {'summary': summary_prompt,
                        'relationship': relationship_prompt}
        self.G = None
        self.user_objectives = self.set_user_objectives(lesson_objectives) if lesson_objectives else {}
        log_level = logging.INFO if verbose else logging.WWARNING
        self.logger = logger_setup(log_level=log_level)

        self.kwargs = kwargs

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_user_objectives(self, objectives: Union[List[str], Dict[str, str]]):
        """
        Sets the lesson objectives provided by the user. If a list is provided, it is converted to a dictionary
        with keys 'Lesson X' corresponding to each lesson in the lesson_range.

        Args:
            objectives (Union[List[str], Dict[str, str]]): The user-provided lesson objectives, either as a list or dictionary.
        """
        if isinstance(objectives, list):
            if len(objectives) != len(self.lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            self.user_objectives = {f'Lesson {i}': obj for i, obj in zip(self.lesson_range, objectives)}
        elif isinstance(objectives, dict):
            if len(objectives) != len(self.lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            self.user_objectives = objectives
        else:
            raise TypeError("Objectives must be provided as either a list or a dictionary.")

    def load_and_process_lessons(self, summary_prompt: str, relationship_prompt: str):
        """
        Loads and processes the lessons by summarizing readings and extracting relationships.
        """
        self.logger.info(f"\nLoading lessons from {self.readings_dir}...")

        # Load documents and lesson objectives
        for lesson_num in self.lesson_range:
            if self.user_objectives:
                self.logger.info('User objectives provided, ignoring provided ')
                lesson_objectives = self.user_objectives.get(f'Lesson {lesson_num}', '')
            else:
                lesson_objectives = extract_lesson_objectives(self.syllabus_path, lesson_num, only_current=True)

            documents = load_lessons(self.readings_dir, lesson_range=range(lesson_num, lesson_num + 1), recursive=self.recursive)

            for document in documents:
                summary = summarize_text(document, prompt=summary_prompt, course_name=self.course_name, llm=self.llm)
                relationships = extract_relationships(summary, lesson_objectives,
                                                      self.course_name, llm=self.llm)

                self.relationship_list.extend(relationships)
                concepts = extract_concepts_from_relationships(relationships)
                self.concept_list.extend(concepts)

        # Process relationships to normalize concepts
        self.logger.info("\nExtracting concepts and relations")
        self.relationship_list = process_relationships(self.relationship_list)
        self.concept_list = list(set(self.concept_list))  # Ensure unique concepts

    def save_intermediate_data(self):
        """
        Saves intermediate data (concepts and relationships) for later use.
        """
        with open(self.output_dir / 'conceptlist.json', 'w') as f:
            json.dump(self.concept_list, f)

        with open(self.output_dir / 'relationship_list.json', 'w') as f:
            json.dump(self.relationship_list, f)

    def build_and_visualize_graph(self, method='leiden'):
        """
        Builds the concept map graph and visualizes it as an interactive HTML file and a word cloud.
        """
        self.logger.info("\nBuilding graph...")
        G = build_graph(self.relationship_list)

        self.logger.info("\nDetecting communities...")
        if not method in ['leiden', 'louvain', 'spectral']:
            raise ValueError("Community detection method not recognized. Please select from 'leiden', 'louvain', or 'spectral'.")
        self.G = detect_communities(G, method=method)  # or "louvain" / "spectral" based on your preference

        output_html_path = self.output_dir / "interactive_concept_map.html"
        visualize_graph_interactive(self.G, output_html_path)

        wordcloud_path = self.output_dir / "concept_wordcloud.png"
        generate_wordcloud(self.concept_list, output_path=wordcloud_path)

    def run_full_pipeline(self):
        """
        Runs the full pipeline for generating a concept map, from loading lessons to producing visual outputs.

        This method orchestrates the entire process of generating a concept map, including:

        1. **Loading and Processing Lessons:**
           - Lessons and associated readings are loaded based on the provided lesson range and reading directories.
           - If user-provided objectives are available, they will be used; otherwise, the method will extract objectives from the syllabus.

        2. **Summarizing and Extracting Relationships:**
           - Each reading is summarized using the provided language model (LLM).
           - Key relationships between concepts are extracted and processed to normalize concept names.
           - If custom prompts are provided, they will be used for summarization and relationship extraction.

        3. **Building the Concept Map:**
           - Constructs a graph based on the extracted relationships.
           - Detects communities within the graph using the specified community detection method.

        4. **Visualizing the Concept Map:**
           - Generates an interactive HTML visualization of the concept map.
           - Creates a word cloud image representing the most frequent concepts.

        Outputs:
            - An interactive HTML file of the concept map is saved to the output directory.
            - A PNG word cloud image is generated and saved to the output directory.

        Note:
            This method uses defaults set during class initialization unless overridden by the provided arguments.

        Raises:
            ValueError: If any part of the process encounters invalid or inconsistent data.
        """
        summary_prompt = self.kwargs.get('summary_prompt', self.prompts['summary'])
        relationship_prompt = self.kwargs.get('relationship_prompt', self.prompts['relationship'])
        method = self.kwargs.get('method', 'leiden')

        self.load_and_process_lessons(summary_prompt=summary_prompt, relationship_prompt=relationship_prompt)
        self.save_intermediate_data()
        self.build_and_visualize_graph(method=method)


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    # env setup
    from pyprojroot.here import here
    load_dotenv()

    # llm chain setup
    from langchain_openai import ChatOpenAI

    # from langchain_community.llms import Ollama
    # Path definitions
    readingDir = Path(os.getenv('readingsDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))
    pdf_syllabus_path = Path(os.getenv('pdf_syllabus_path'))

    projectDir = here()

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
    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0,

    #     )

    builder = ConceptMapBuilder(
        readings_dir=readingDir,
        project_dir=projectDir,
        syllabus_path=syllabus_path,
        llm=llm,
        course_name="American Politics",
        lesson_range=range(17),
        recursive=True,
        verbose=True
    )

    builder.run_full_pipeline()
