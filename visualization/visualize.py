import graphviz
import os
import subprocess

def get_graphviz_path():
    """Find Graphviz installation path and verify it works"""
    # First check if dot is already in PATH
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
        return None  # Graphviz is already in PATH
    except:
        pass

    # Common Windows installation paths
    possible_paths = [
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
        r"C:\Graphviz\bin",
        r"E:\windows_10_cmake_Release_Graphviz-12.2.1-win64\Graphviz-12.2.1-win64\bin"
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            # If it's a directory, check for dot.exe
            if os.path.isdir(path):
                dot_path = os.path.join(path, 'dot.exe')
                if os.path.exists(dot_path):
                    return path
            # If it's a file, return its directory
            elif os.path.isfile(path):
                return os.path.dirname(path)
    
    return None

# Set Graphviz path if needed
GRAPHVIZ_PATH = get_graphviz_path()
if GRAPHVIZ_PATH:
    print(f"Found Graphviz at: {GRAPHVIZ_PATH}")
    os.environ["PATH"] = GRAPHVIZ_PATH + os.pathsep + os.environ["PATH"]
else:
    print("Graphviz not found in common locations. Please ensure it's installed and in PATH.")

# MARKDOWN BACKUP
def tree_to_markdown(tree, prefix="", is_last=True):
    if not isinstance(tree, dict):
        return f"{prefix}{'└── ' if is_last else '├── '}{tree}\n"

    question = list(tree.keys())[0]
    yes_answer, no_answer = tree[question]
    result = f"{prefix}{'└── ' if is_last else '├── '}{question}\n"
    new_prefix = prefix + ("    " if is_last else "│   ")
    result += tree_to_markdown(yes_answer, new_prefix, False)
    result += tree_to_markdown(no_answer, new_prefix, True)
    return result

def visualize_tree(tree):
    tree_str = tree_to_markdown(tree)
    legend = """
Legend:
├── : Branch continues
└── : Last branch at this level
"""
    return tree_str + legend


# GRAPHVIZ VISUALIZATION WITH READABILITY ENHANCEMENTS
def visualize_tree_graphviz(tree, filename="decision_tree"):
    try:
        # Verify Graphviz installation
        try:
            subprocess.run(['dot', '-V'], capture_output=True, check=True)
        except Exception as e:
            raise Exception("Graphviz is not installed or not found in PATH. Please install Graphviz and ensure it's in your system PATH.")

        dot = graphviz.Digraph(comment='Decision Tree', engine='dot')
        
        # Global visual settings for better readability
        dot.attr(rankdir='TB')
        dot.attr('node',
                 shape='box',
                 style='filled',
                 fontname='Helvetica',
                 fontsize='14',
                 margin='0.3,0.2',
                 width='0',
                 height='0')
        dot.attr('edge',
                 fontname='Helvetica',
                 fontsize='12')
        dot.attr(label='Decision Tree', fontsize='18', fontname='Helvetica Bold', labelloc="t")

        node_counter = {"count": 0}

        def add_nodes_edges(tree, parent_id=None, edge_label=""):
            if not isinstance(tree, dict):
                node_id = f"leaf_{node_counter['count']}"
                node_counter['count'] += 1
                dot.node(node_id, str(tree),
                         shape='box',
                         fillcolor='lightblue',
                         fontcolor='black')
                if parent_id:
                    dot.edge(parent_id, node_id, label=edge_label)
                return

            question = list(tree.keys())[0]
            yes_answer, no_answer = tree[question]

            node_id = f"node_{node_counter['count']}"
            node_counter['count'] += 1
            dot.node(node_id, question,
                     shape='box',
                     fillcolor='lightgreen',
                     fontcolor='black')

            if parent_id:
                dot.edge(parent_id, node_id, label=edge_label)

            add_nodes_edges(yes_answer, node_id, "Yes")
            add_nodes_edges(no_answer, node_id, "No")

        add_nodes_edges(tree)

        # Render the graph to PNG
        dot.render(filename, format='png', view=True, cleanup=True)
        return dot

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None