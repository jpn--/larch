import os.path
from glob import glob

import nbformat as nbf

# Collect a list of all notebooks in the docs folder
notebooks = glob(
    os.path.join(os.path.dirname(__file__), "..", "**", "*.ipynb"),
    recursive=True,
)

# Text to look for in adding tags
text_search_dict = {
    "# TEST": "remove_cell",  # Remove the whole cell
    "# HIDDEN": "remove_cell",  # Remove the whole cell
    "# NO CODE": "remove_input",  # Remove only the input
    "# HIDE CODE": "hide_input",  # Hide the input w/ a button to show
}

# Search through each notebook and look for th text, add a tag if necessary
for ipath in notebooks:
    if "/_build/" in ipath:
        continue
    touch = False
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        cell_tags = cell.get("metadata", {}).get("tags", [])
        for key, val in text_search_dict.items():
            if key in cell["source"]:
                if val not in cell_tags:
                    cell_tags.append(val)
                    touch = True
        if len(cell_tags) > 0:
            cell["metadata"]["tags"] = cell_tags
    if touch:
        print(f"hiding test cells in {ipath}")
        nbf.write(ntbk, ipath)
    else:
        print(f"no changes in {ipath}")
