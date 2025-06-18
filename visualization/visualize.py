import graphviz
import subprocess

def visualize_tree(tree):
    """
    Create a Graphviz visualization of the decision tree.
    
    Parameters:
    -----------
    tree : dict
        The decision tree structure
        
    Returns:
    --------
    graphviz.Digraph
        A Graphviz object that can be displayed in Streamlit
    """
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
                 height='0',
                 fillcolor='#E8F5E9',  # Light green background
                 fontcolor='black')    # Black text
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
                         fillcolor='#E3F2FD',  # Light blue background
                         fontcolor='black')     # Black text
                if parent_id:
                    dot.edge(parent_id, node_id, label=edge_label)
                return

            question = list(tree.keys())[0]
            yes_answer, no_answer = tree[question]

            node_id = f"node_{node_counter['count']}"
            node_counter['count'] += 1
            dot.node(node_id, question,
                     shape='box',
                     fillcolor='#E8F5E9',  # Light green background
                     fontcolor='black')     # Black text

            if parent_id:
                dot.edge(parent_id, node_id, label=edge_label)

            add_nodes_edges(yes_answer, node_id, "Yes")
            add_nodes_edges(no_answer, node_id, "No")

        add_nodes_edges(tree)
        return dot

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

# Keep the existing functions for reference
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

def visualize_tree_markdown(tree):
    tree_str = tree_to_markdown(tree)
    legend = """
Legend:
├── : Branch continues
└── : Last branch at this level
"""
    return tree_str + legend

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