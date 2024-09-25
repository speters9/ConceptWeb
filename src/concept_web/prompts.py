"""Prompts for llm summarizing and relationship extraction"""

summary_prompt = """You are a professor specializing in {course_name}
                    You will be given a text and asked to summarize this text in light of your expertise.
                    Summarize the following text: \n {text}"""

relationship_prompt = """You are a professor specializing in {course_name}.
                        You are instructing an introductory undergraduate {course_name} class.
                        You will be mapping relationships between the concepts this class addresses.
                        The objectives for this lesson are: \n {objectives} \n

                        From the following text for this lesson, extract the key concepts and the relationships between them.
                        Identify the key concepts and then explain how each relates to the others.
                        \n
                        {text}
                        \n

                        Extract the most important and generally applicable key concepts and themes from the above summary.
                        Focus on high-level concepts or overarching themes relevant to an undergraduate {course_name} course and the lesson objectives.
                        Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                        Avoid overly specific or narrow topics.

                        Provide the relationships between each concept with the other discovered concepts in the format:
                            "relationships": [
                              ["Concept 1", "relationship_type", "Concept 2"],
                              ["Concept 1", "relationship_type", "Concept 3"],
                              ...
                            ]

                        If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of {course_name}, \
                        return "None" in the "relationship_type" field.

                        Because you are comparing a lot of concepts, the json may be long. That's fine.

                        Ensure results are returned in a valid json.
                        """

no_objective_relationship_prompt = """You are a political science professor specializing in {course_name}.
                        You are instructing an introductory undergraduate {course_name} class.
                        You will be mapping relationships between the concepts this class addresses.

                        From the following text for this lesson, extract the key concepts and the relationships between them.
                        Identify the key concepts and then explain how each relates to the others.
                        \n
                        {text}
                        \n

                        Extract the most important and generally applicable key concepts and themes from the above summary.
                        Focus on high-level concepts or overarching themes relevant to an undergraduate {course_name} course.
                        Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                        Avoid overly specific or narrow topics.

                        Provide the relationships between each concept with the other discovered concepts in the format:
                            "relationships": [
                              ["Concept 1", "relationship_type", "Concept 2"],
                              ["Concept 1", "relationship_type", "Concept 3"],
                              ...
                            ]

                        If there is no meaningful relationship from the standpoint of your expertise as a professor of {course_name}, \
                        return "None" in the "relationship_type" field.

                        Because you are comparing a lot of concepts, the json may be long. That's fine.

                        Ensure results are returned in a valid json.
                        """
