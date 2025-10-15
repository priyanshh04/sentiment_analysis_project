# Create the main project structure and configuration
import os
import pandas as pd
import numpy as np

# Create project directory structure
project_structure = {
    'sentiment_analysis_project/': {
        'data/': ['raw/', 'processed/', 'models/'],
        'src/': ['preprocessing/', 'models/', 'evaluation/', 'utils/'],
        'notebooks/': [],
        'configs/': [],
        'tests/': [],
        'requirements/': []
    }
}

def create_directory_structure(base_path='.'):
    """Create the project directory structure"""
    dirs_created = []
    
    for main_dir, subdirs in project_structure.items():
        main_path = os.path.join(base_path, main_dir)
        os.makedirs(main_path, exist_ok=True)
        dirs_created.append(main_path)
        
        if isinstance(subdirs, dict):
            for subdir, subsubdirs in subdirs.items():
                sub_path = os.path.join(main_path, subdir)
                os.makedirs(sub_path, exist_ok=True)
                dirs_created.append(sub_path)
                
                for subsubdir in subsubdirs:
                    subsub_path = os.path.join(sub_path, subsubdir)
                    os.makedirs(subsub_path, exist_ok=True)
                    dirs_created.append(subsub_path)
        else:
            for subdir in subdirs:
                sub_path = os.path.join(main_path, subdir)
                os.makedirs(sub_path, exist_ok=True)
                dirs_created.append(sub_path)
    
    return dirs_created

# Create the directory structure
dirs_created = create_directory_structure()
print("PROJECT DIRECTORY STRUCTURE CREATED:")
print("=" * 50)
for dir_path in sorted(dirs_created):
    print(f"📁 {dir_path}")

print(f"\nTotal directories created: {len(dirs_created)}")
print("\nNext: Creating Python files with complete implementation...")