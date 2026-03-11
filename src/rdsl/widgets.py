try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display
except ImportError:
    msg = "display_functional_groups requires ipywidgets. Install it with 'pip install ipywidgets'."
    raise ImportError(msg) from None

from rdkit import Chem
from rdsl.functional_groups import get_functional_group_matches
from rdsl.highlight import highlight_atoms

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

def display_functional_groups(mol: Chem.Mol, show_overshadowed: bool = False):
    """
    Creates a Jupyter widget that displays the molecule with functional group highlights
    and a selectable list of functional groups.

    Args:
        mol: The RDKit molecule to display.
        show_overshadowed: Whether to show overshadowed groups by default.
    """
    # 0. Prepare Molecule (Force consistent 2D coordinates)
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)

    # 1. Get matches
    all_df = get_functional_group_matches(mol, include_overshadowed=True)
    primary_df = get_functional_group_matches(mol, include_overshadowed=False)

    # Mark overshadowed
    primary_keys = set(zip(primary_df["name"], primary_df["atom_ids"]))
    all_df["is_overshadowed"] = all_df.apply(
        lambda x: (x["name"], x["atom_ids"]) not in primary_keys, axis=1
    )

    # 2. Sort according to requirements
    match_sort_order = {
        "functional_group": 0,
        "cyclic": 1,
        "biological": 2,
        "overshadowed": 3,
    }
    
    def get_sort_rank(row):
        if row["is_overshadowed"]:
            return match_sort_order["overshadowed"]
        return match_sort_order.get(row["group"], 99)

    all_df["sort_rank"] = all_df.apply(get_sort_rank, axis=1)
    # Sort by rank, then name
    all_df = all_df.sort_values(["sort_rank", "name"]).reset_index(drop=True)

    # 3. State
    initial_selection = None
    if not all_df.empty:
        if not show_overshadowed:
            primary_indices = all_df[~all_df["is_overshadowed"]].index
            initial_selection = primary_indices[0] if not primary_indices.empty else 0
        else:
            initial_selection = 0

    state = {
        "selected_idx": initial_selection,
        "show_overshadowed": show_overshadowed,
        "filtered_indices": []  # Indices into all_df
    }

    # 4. Widgets
    mol_output = widgets.Output(layout=widgets.Layout(width="550px", height="450px", display="flex", align_items="center", justify_content="center"))
    list_header = widgets.HTML("<h2 style='font-family: sans-serif; margin-bottom: 10px;'>Substructures</h2>")
    
    # We'll use a container for buttons to allow scrolling
    buttons_layout = widgets.VBox(layout=widgets.Layout(max_height="400px", overflow_y="auto"))
    list_container = widgets.VBox([list_header, buttons_layout], layout=widgets.Layout(width="350px"))
    
    toggle_btn = widgets.Button(
        description="Toggle overshadowed ⓘ",
        layout=widgets.Layout(width="100%", height="40px", margin="10px 0 0 0"),
    )
    toggle_btn.add_class("fg-toggle-btn")

    css = widgets.HTML("""
    <style>
        .fg-btn {
            border: none !important;
            text-align: left !important;
            padding-left: 15px !important;
            font-family: sans-serif !important;
            font-size: 14px !important;
            border-radius: 4px !important;
            margin: 1px 0 !important;
            transition: all 0.2s !important;
            flex-shrink: 0 !important;
            min-height: 35px !important;
        }
        
        .fg-btn-functional_group { background-color: #e7f1ff !important; color: #0056b3 !important; }
        .fg-btn-cyclic { background-color: #e2e3e5 !important; color: #383d41 !important; }
        .fg-btn-biological { background-color: #d4edda !important; color: #155724 !important; }
        .fg-btn-overshadowed { background-color: #fff3cd !important; color: #856404 !important; }
        
        .fg-btn-selected.fg-btn-functional_group { background-color: #007bff !important; color: white !important; }
        .fg-btn-selected.fg-btn-cyclic { background-color: #6c757d !important; color: white !important; }
        .fg-btn-selected.fg-btn-biological { background-color: #28a745 !important; color: white !important; }
        .fg-btn-selected.fg-btn-overshadowed { background-color: #ffc107 !important; color: white !important; }
        
        .fg-toggle-btn {
            background-color: #6c757d !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: bold !important;
        }
    </style>
    """)

    # 5. Logic
    def update_molecule_display():
        with mol_output:
            clear_output(wait=True)
            
            # Use fixed size SVG for stable alignment
            width, height = 550, 450
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            
            if state["selected_idx"] is not None:
                row = all_df.iloc[state["selected_idx"]]
                highlight_atoms = [int(i) for i in row["atom_ids"]]
                drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
            else:
                drawer.DrawMolecule(mol)
            
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            from IPython.display import SVG
            display(SVG(svg))

    def refresh_button_styles():
        """Updates the CSS classes of existing buttons without re-rendering the list."""
        for btn in buttons_layout.children:
            idx = getattr(btn, "_match_idx", None)
            if idx is None:
                continue
            
            # Update the selected class
            classes = list(btn._dom_classes)
            if state["selected_idx"] == idx:
                if "fg-btn-selected" not in classes:
                    btn.add_class("fg-btn-selected")
            else:
                if "fg-btn-selected" in classes:
                    btn._dom_classes = tuple(c for c in classes if c != "fg-btn-selected")

    def update_list_structure():
        """Rebuilds the button list (e.g. when filters change). This WILL reset scroll."""
        mask = [True] * len(all_df)
        if not state["show_overshadowed"]:
            mask = ~all_df["is_overshadowed"]
        
        visible_df = all_df[mask]
        state["filtered_indices"] = list(visible_df.index)
        
        buttons = []
        for idx in state["filtered_indices"]:
            row = all_df.iloc[idx]
            btn = widgets.Button(
                description=row["name"].capitalize(),
                layout=widgets.Layout(width="100%", height="35px"),
            )
            # Store the index on the button object for future reference
            btn._match_idx = idx
            btn.add_class("fg-btn")
            
            # Determine color class
            btn_type = "overshadowed" if row["is_overshadowed"] else row["group"]
            btn.add_class(f"fg-btn-{btn_type}")
            
            # Callback
            def on_click_handler(b, target_idx=idx):
                state["selected_idx"] = target_idx
                update_molecule_display()
                refresh_button_styles()
            
            btn.on_click(on_click_handler)
            buttons.append(btn)
        
        buttons_layout.children = buttons
        refresh_button_styles()

        # Ensure toggle button is always at the bottom of the right panel
        if list_container.children[-1] != toggle_btn:
            list_container.children = list(list_container.children) + [toggle_btn]

    def on_toggle_clicked(b):
        state["show_overshadowed"] = not state["show_overshadowed"]
        # If current selected is overshadowed and we hide them, reset selection
        if not state["show_overshadowed"] and state["selected_idx"] is not None:
            if all_df.iloc[state["selected_idx"]]["is_overshadowed"]:
                # Try to find first non-overshadowed
                primary_indices = all_df[~all_df["is_overshadowed"]].index
                state["selected_idx"] = primary_indices[0] if not primary_indices.empty else None
        
        update_list_structure()
        update_molecule_display()

    toggle_btn.on_click(on_toggle_clicked)

    # Initial update
    update_list_structure()
    update_molecule_display()

    # Layout
    arrow = widgets.HTML("<div style='font-size: 32px; color: #ccc; margin: auto 40px;'>&rsaquo;</div>")
    main_box = widgets.HBox([mol_output, arrow, list_container], layout=widgets.Layout(align_items="center"))
    
    display(css)
    return main_box
