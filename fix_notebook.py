import json
import sys

def fix_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_made = []

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            original_source = source

            # Change OneCycleLR max_lr from LR*2 to LR*1
            if 'OneCycleLR' in source and 'max_lr=LR*2' in source:
                source = source.replace('max_lr=LR*2', 'max_lr=LR*1')
                changes_made.append("Changed OneCycleLR max_lr from LR*2 to LR*1")

            # Add forward pass NaN check in train_epoch function
            if 'def train_epoch' in source:
                lines = source.splitlines(keepends=True)
                new_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    new_lines.append(line)
                    # Look for the model call - be flexible with whitespace
                    if 'bl,sl,unc,aux = model(x40,x100,x200,x400)' in line.strip():
                        # Insert after this line
                        new_lines.append('    # Check for NaNs in forward pass\n')
                        new_lines.append('    for name, tensor in [(\'binary_logits\', bl), (\'subtype_logits\', sl), (\'uncertainty\', unc)]:\n')
                        new_lines.append('        if not torch.isfinite(tensor).all():\n')
                        new_lines.append("            print(f'  ⚠ NaN in forward pass: {name} at step {step}')\n")
                    i += 1
                source = ''.join(new_lines)
                if source != original_source:
                    changes_made.append("Added forward pass NaN check in train_epoch")

            # Update the cell if changes were made
            if source != original_source:
                cell['source'] = source.splitlines(keepends=True)

    if changes_made:
        print("Changes made:")
        for change in changes_made:
            print(f"  - {change}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed notebook saved to {notebook_path}")
    else:
        print("No changes made - patterns not found")

if __name__ == '__main__':
    fix_notebook('MAGNET_Final_Fixed.ipynb')