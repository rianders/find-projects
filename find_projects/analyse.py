import os
import json
import click
import ollama
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import tempfile
import asyncio
import aiofiles
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# Assuming your Ollama API key is stored in an environment variable called OLLAMA_API_KEY
ollama_api_key = os.getenv("OLLAMA_API_KEY")

# Initialize Ollama API client (using host and timeout if needed)
ollama_client = ollama.Client(host='http://localhost:11434')

# Path where JSON will be stored
JSON_FILE_PATH = "projects_info.json"

# Function to read README or pyproject.toml files
def read_project_info(directory):
    possible_files = ["README.md", "readme.md", "README.txt", "readme.txt", "pyproject.toml"]
    for file_name in possible_files:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                return file.read()
    return None

# Function to read .gitignore and parse ignored patterns
def get_gitignore_patterns(directory):
    gitignore_path = os.path.join(directory, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return []
    
    with open(gitignore_path, "r") as file:
        patterns = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    return patterns

# Function to check if a file or directory should be ignored based on .gitignore patterns
def should_ignore(path, patterns):
    from fnmatch import fnmatch
    return any(fnmatch(path, pattern) for pattern in patterns)

# Function to generate a tree-like structure of the project
async def generate_project_tree(directory, patterns, prefix=""):
    tree = []
    async for entry in aiofiles.os.scandir(directory):
        if should_ignore(entry.name, patterns):
            continue
        
        tree.append(f"{prefix}{entry.name}")
        if entry.is_dir():
            subtree = await generate_project_tree(entry.path, patterns, prefix=prefix + "|   ")
            tree.extend(subtree)
    return tree

# Function to scan each project directory inside a parent directory
async def scan_project_directories(parent_directory, append=False):
    project_directories = [entry.path for entry in os.scandir(parent_directory) if entry.is_dir()]
    results = []

    if not append:
        # If not appending, start with an empty JSON file
        if os.path.exists(JSON_FILE_PATH):
            os.remove(JSON_FILE_PATH)

    with tqdm(total=len(project_directories), desc="Scanning project directories") as pbar:
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(None, scan_directory, project_dir) for project_dir in project_directories]
        for future in asyncio.as_completed(futures):
            project_data = await future
            results.append(project_data)
            update_json(project_data)
            pbar.update(1)

    return results

# Function to scan project directory
def scan_directory(directory):
    try:
        patterns = get_gitignore_patterns(directory)
        project_tree = generate_project_tree_sync(directory, patterns)
        
        # Write project tree to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
            for line in project_tree:
                temp_file.write(line + "\n")
            temp_file_path = temp_file.name
        
        readme_content = read_project_info(directory)
        
        if not readme_content:
            return {
                "title": os.path.basename(directory),
                "directory": directory,
                "error": "README or similar file not found"
            }
        
        # Ask Ollama to analyze the project
        response = ollama_client.chat(
            model='llama3.2:1b',
            messages=[
                {
                    'role': 'user',
                    'content': f"Analyze the following project description and provide a summary including project name, technology stack, and its purpose:\n{readme_content}\n\nAdditionally, here is the project's directory structure:\n{open(temp_file_path).read()}"
                }
            ]
        )
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return {
            "title": os.path.basename(directory),
            "directory": directory,
            "summary": response.get('message', {}).get('content', 'No summary provided'),
            "tech_stack": response.get('tech_stack', 'No tech stack identified'),
            "purpose": response.get('purpose', 'No purpose identified')
        }
    except Exception as e:
        return {
            "title": os.path.basename(directory),
            "directory": directory,
            "error": f"An error occurred: {str(e)}"
        }

# Synchronous function for generating a project tree
def generate_project_tree_sync(directory, patterns, prefix=""):
    tree = []
    with os.scandir(directory) as it:
        for entry in it:
            if should_ignore(entry.name, patterns):
                continue
            
            tree.append(f"{prefix}{entry.name}")
            if entry.is_dir():
                subtree = generate_project_tree_sync(entry.path, patterns, prefix=prefix + "|   ")
                tree.extend(subtree)
    return tree

# Function to update the JSON file with new project information
def update_json(project_data):
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r") as file:
            projects = json.load(file)
    else:
        projects = []

    projects.append(project_data)

    with open(JSON_FILE_PATH, "w") as file:
        json.dump(projects, file, indent=4)

# Click CLI for searching and analyzing projects
@click.command()
@click.argument('parent_directory', type=click.Path(exists=True))
@click.option('--append', is_flag=True, help="Append to the existing JSON file instead of replacing it.")
def analyze_projects(parent_directory, append):
    """Scans all project directories inside the specified parent directory and uses Ollama to determine project details."""
    asyncio.run(scan_project_directories(parent_directory, append=append))
    click.echo("Project information updated in JSON.")

if __name__ == "__main__":
    analyze_projects()