
import nbformat
import os

def update_notebook(path):
    print(f"Updating {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    changed = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            original_source = cell.source
            new_source = original_source
            
            # Replace super_agent_env imports
            if 'from super_agent_env import' in new_source:
                new_source = new_source.replace('from super_agent_env import', 'from environments import')
                changed = True
            
            # Replace meta_agent_env imports
            if 'from meta_agent_env import' in new_source:
                new_source = new_source.replace('from meta_agent_env import', 'from environments import')
                changed = True
                
            # Replace meta_agent_enviroment imports (typo seen in diff)
            if 'from meta_agent_enviroment import' in new_source:
                new_source = new_source.replace('from meta_agent_enviroment import', 'from environments import')
                changed = True

            # Replace walkforwardclass imports
            if 'from walkforwardclass import' in new_source:
                new_source = new_source.replace('from walkforwardclass import', 'from walk_forward_validation import')
                changed = True
            
            if new_source != original_source:
                print(f"  Modified cell:\n{'-'*20}\n{original_source}\n{'-'*20}\nTO\n{'-'*20}\n{new_source}\n{'-'*20}")
                cell.source = new_source
    
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"✓ Saved updates to {path}")
    else:
        print(f"  No changes needed for {path}")

if __name__ == "__main__":
    notebooks = [
        '02_super_meta_agents.ipynb',
        '03_super_meta_agents_walk_forward.ipynb'
    ]
    
    for nb in notebooks:
        if os.path.exists(nb):
            update_notebook(nb)
        else:
            print(f"Notebook not found: {nb}")
