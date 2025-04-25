from organisms import NEATOrganism
from graphviz import Digraph

class Node:
    def __init__(self, name="", label="", color=None):
        self.parent = None
        self.name = name
        self.label = label
        self.color = color
        self.bgcolor = None

def get_tree(population: list[NEATOrganism], best_agent=None):
    tree = {}
    if best_agent is None:
        best_agent = population[0]

    for agent in population:
        heredity = []
        current = agent
        architecture = str(agent).replace("NEATOrganism::", "")
        agent_node = Node(
            name=f"{id(agent)}",
            label=f"Input -> {str(agent).replace('NEATOrganism::','')}{' -> ' if architecture != '' else ''}Output")
        if agent == best_agent:
            agent_node.bgcolor = "green"
        agent_node.color = "green"

        current_node = agent_node
        while current.parent != None:
            parent = current.parent
            architecture = str(parent).replace("NEATOrganism::", "")
            parent_node = Node(
                name=f"{id(parent)}",
                label=f"Input -> {architecture}{' -> ' if architecture != '' else ''}Output",
                color="green" if parent in population else None
            )
            current_node.parent = parent_node
            heredity.append(parent_node)
            current = current.parent
            current_node = parent_node
        
        add_to = tree
        for ancestor in reversed(heredity):
            if ancestor not in add_to.keys() or add_to[ancestor] == None:
                add_to[ancestor] = {}
            add_to = add_to[ancestor]
        add_to[agent_node] = None
    return tree

def traverse_tree(level: dict, d: Digraph, root: Node | None=None, memo=set()):
    if root != None:
        d.node(
            root.name,
            root.label,
            color=root.color,
            fillcolor=root.bgcolor,
            style="filled" if root.bgcolor != None else None
        )
        if level == None:
            return
        for child in level.keys():
            if (root.name, child.name) not in memo:
                d.edge(root.name, child.name)
                memo.add((root.name, child.name))
    for node, child_dict in level.items():
        traverse_tree(child_dict, d, root=node, memo=memo)

def generate_genealogy(output_loc: str, population: list[NEATOrganism], best_agent=None, graph_attr={}):
    d = Digraph(graph_attr=graph_attr)
    tree = get_tree(population=population, best_agent=best_agent)
    traverse_tree(tree, d)
    d.save(output_loc)
    d.render(output_loc, format='svg', view=False)
