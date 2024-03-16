from langchain_core.prompts import PromptTemplate


p1 = PromptTemplate.from_template(
    """
The program repository {repo} is coded in {langs}.
About the repository: {readme}
However, there is an issue in the code:
{issue}
What language do you think will the solution be in? Choose from {langs} and ONLY output a single word.
"""
)

p2 = PromptTemplate.from_template(
    """You are debugging an issue in {repo} programmed in {langs}.
Issue: {issue}
The following files are present: {files}
Which file may contain the relevant functions/code for solving this issue? Write the path of the relevant file.
"""
)

p3 = PromptTemplate.from_template(
    """You need to fix an issue in {repo}, {readme} in {file_path}
    Issue: {issue}
    Code blocks: {code} 
    Refactor the code to rectify this issue and review your code to ensure its validity.
"""
)

p4 = PromptTemplate.from_template("""
You are an expert linux sysadmin. Your current location is {repo}:
The following subdir structure is present:
{subdirs}
Describe the structure in natural language. Make sure to mention filenames and paths and other details.
""")