from kaggle.api.kaggle_api_extended import KaggleApi
import networkx as nx
import re
import os
import matplotlib.pyplot as plt

DELETE_UNUSED = True
KEEP_INTERMEDIATE = False
FORCE_INTERMEDIATE = False
NUM_WORKERS = 1
DATASETS_DIR = './datasets/'

KAGGLE_API = KaggleApi()
KAGGLE_API.authenticate()

class SizeTracker():
    def __init__(self):
        self.current = 0
        self.max = 0

    def add(self, v):
        self.current += v
        self.max = max(self.max, self.current)

    def sub(self, v):
        self.current -= v

# GRAPH ---------------------------------------------------------------------

class Node():
    def __init__(self, children_nodes):
        pass

    def create(self, dry=False):
        raise Exception('ABC')

    def delete(self, dry=False):
        raise Exception('ABC')

    def exists(self):
        raise Exception('ABC')

    def get_artifact_path(self):
        raise Exception('ABC')

    def get_artifact_size(self):
        raise Exception('ABC')

    def get_max_disk_space_usage(self):
        raise Exception('ABC')

    def get_additional_required_disk_space_usage(self):
        raise Exception('ABC')

    def get_additional_download_size(self):
        return 0

def append_from_file_to_file(from_path, to_path):
    CHUNK_SIZE = 1024 * 128
    with open(from_path, 'rb') as from_file:
        with open(to_path, 'ab') as to_file:
            to_path.write(from_path.read(CHUNK_SIZE))

class KaggleDownloadNode(Node):
    def __init__(self, destination, url, combine='cat'):
        if combine != 'cat':
            raise Exception('Unimplemented')

        self.destination = destination
        self.url = url
        self.combine = combine
        self.dataset = re.search(r"^https://www.kaggle.com/datasets/(.*)", self.url).group(1)

        print(f'Discovering Kaggle dataset {self.dataset}.')

        self.files = sorted(KAGGLE_API.dataset_list_files(self.dataset).files)
        self.destination_size = self.calculate_destination_size()

    def create(self, dry=False):
        print(f'Download Kaggle dataset {self.dataset} to {self.destination}. Files combined with `{self.combine}`.')
        if not dry:
            if self.exists():
                print(f'Dataset already present.')
            else:
                print(f'Dataset not present. Downloading...')
                for i, file in enumerate(self.files):
                    temp_path = destination + f'.part{i}'
                    print(f'Downloading file {i+1} / {len(self.files)}...')
                    KAGGLE_API.download_dataset_file(dataset=self.dataset, file_name=file, path=temp_path, force=False, quiet=True)
                    if i == 0:
                        os.rename(temp_path, self.destination)
                    else:
                        append_from_file_to_file(temp_path, self.destination)
                        os.remove(temp_path)

    def delete(self, dry=False):
        print(f'Delete file {self.destination}.')
        if not dry:
            os.remove(self.destination)

    def exists(self):
        return os.path.exists(self.destination) and os.path.getsize(self.destination) == self.destination_size

    def get_artifact_path(self):
        return self.destination

    def get_artifact_size(self):
        return self.destination_size

    def calculate_destination_size(self):
        total_size = 0
        for file in self.files:
            total_size += file.totalBytes
        return total_size

    def get_max_disk_space_usage(self):
        return self.destination_size

    def get_additional_required_disk_space_usage(self):
        # first file we can move
        max_subsequent_file_size = max([0] + [f.totalBytes for f in self.files[1:]])
        try:
            return self.destination_size - os.path.getsize(self.destination) + max_subsequent_file_size
        except:
            return self.destination_size + max_subsequent_file_size

    def get_additional_download_size(self):
        return self.destination_size

    def __hash__(self):
        return hash((self.destination, self.url, self.combine))

    def __eq__(self, other):
        return self.destination == other.destination and self.url == other.url and self.combine == other.combine

class CatNode(Node):
    def __init__(self, destination, children_nodes):
        self.destination = destination
        self.children_nodes = sorted(children_nodes, key=lambda x: x.get_artifact_path())
        self.destination_size = sum(child.get_artifact_size() for child in self.children_nodes)

    def create(self, dry=False):
        sources = []
        for child in self.children_nodes:
            path = child.get_artifact_path()
            if not path.endswith('.binpack'):
                raise Exception('Expected a child with .binpack dataset artifact.')
            sources.append(path)
        print(f'Concatenate {len(sources)} files into {self.destination}.')
        if not dry:
            if self.exists():
                print(f'Dataset already present.')
            else:
                print(f'Dataset not present. Concatenating')
                with open(self.destination, 'wb') as outfile:
                    pass
                for child in self.children_nodes:
                    child_path = child.get_artifact_path()
                    print(f'Appending {child_path}')
                    append_from_file_to_file(child_path, self.destination)

    def delete(self, dry=False):
        print(f'Delete file {self.destination}.')
        if not dry:
            os.remove(self.destination)

    def exists(self):
        return os.path.exists(self.destination) and os.path.getsize(self.destination) == self.destination_size

    def get_artifact_path(self):
        return self.destination

    def get_artifact_size(self):
        return self.destination_size

    def get_max_disk_space_usage(self):
        return self.get_artifact_size()

    def get_additional_required_disk_space_usage(self):
        try:
            return self.destination_size - os.path.getsize(self.destination)
        except:
            return self.destination_size

    def __hash__(self):
        return hash((self.destination,))

    def __eq__(self, other):
        return self.destination == other.destination

class InterleaveNode(Node):
    def __init__(self, destination, children_nodes):
        self.destination = destination
        self.children_nodes = children_nodes
        self.destination_size = sum(child.get_artifact_size() for child in children_nodes)

    def create(self, dry=False):
        sources = []
        for child in self.children_nodes:
            path = child.get_artifact_path()
            if not path.endswith('.binpack'):
                raise Exception('Expected a child with .binpack dataset artifact.')
            sources.append(path)
        print(f'Interleave {len(sources)} files into {self.destination}.')
        if not dry:
            if self.exists():
                print(f'Dataset already present.')
            else:
                raise Exception('Unimplemented') # TODO: set up the tools branch

    def delete(self, dry=False):
        print(f'Delete file {self.destination}.')
        if not dry:
            os.remove(self.destination)

    def exists(self):
        return os.path.exists(self.destination) and os.path.getsize(self.destination) == self.destination_size

    def get_artifact_path(self):
        return self.destination

    def get_artifact_size(self):
        return self.destination_size

    def get_max_disk_space_usage(self):
        return self.get_artifact_size()

    def get_additional_required_disk_space_usage(self):
        try:
            return self.destination_size - os.path.getsize(self.destination)
        except:
            return self.destination_size

    def __hash__(self):
        return hash((self.destination,))

    def __eq__(self, other):
        return self.destination == other.destination

class DatasetGraph():
    def __init__(self, datasets_dir):
        self.graph = nx.DiGraph()
        self.datasets_dir = datasets_dir

    def add_dataset_definition(self, dataset_def):
        dataset_def(self.graph, self.datasets_dir)

    def draw(self):
        pos = nx.spring_layout(self.graph, seed=1234)
        labels = dict()
        for node in self.graph.nodes:
            labels[node] = node.get_artifact_path()
        nx.draw_networkx(self.graph, pos, labels=labels, label='Depencency graph')
        plt.show()

    def traverse(self, on_create, on_delete):
        visited = set()
        finished = set()

        def visit(node):
            if node in visited:
                return

            visited.add(node)

            # only descend to children if they are actually required
            if not node.exists():

                edges = self.graph.edges(node)
                children_nodes = [j for i, j in edges]

                for child in children_nodes:
                    visit(child)

                on_create(node)
                finished.add(node)

            if not KEEP_INTERMEDIATE:
                for child in children_nodes:
                    # If all parents have been finished we don't need this child anymore
                    if all(i in finished for i, j in self.graph.in_edges(child)):
                        on_delete(child)

        for node in self.graph.nodes:
            # Only expand top level nodes.
            # We don't want to process unneeded intermediates. Unless explicitely specified.
            if FORCE_INTERMEDIATE or len(self.graph.in_edges(node)) == 0:
                visit(node)

    def create(self, dry=False):
        self.traverse(
            on_create=lambda n: n.create(dry),
            on_delete=lambda n: n.delete(dry)
        )

    def get_additional_required_disk_space_usage(self):
        additional_usage = SizeTracker()
        additional_download_size = 0

        def on_create(node):
            nonlocal additional_usage
            nonlocal additional_download_size
            additional_download_size += node.get_additional_download_size()
            additional_usage.add(node.get_additional_required_disk_space_usage())

        def on_delete(node):
            nonlocal additional_usage
            nonlocal additional_download_size
            additional_usage.sub(node.get_artifact_size())

        self.traverse(on_create=on_create, on_delete=on_delete)

        return additional_usage, additional_download_size

# GRAPH END -----------------------------------------------------------------

# DSL -----------------------------------------------------------------------

def interleave(destination, sources):
    def visit(graph, root_dir):
        children_nodes = []
        for source in sources:
            children_nodes.append(source(graph, root_dir))
        node = InterleaveNode(os.path.join(root_dir, destination), children_nodes)
        graph.add_node(node)
        for child_node in children_nodes:
            graph.add_edge(node, child_node)
        return node
    return visit

def cat(destination, sources):
    def visit(graph, root_dir):
        children_nodes = []
        for source in sources:
            children_nodes.append(source(graph, root_dir))
        node = CatNode(os.path.join(root_dir, destination), children_nodes)
        graph.add_node(node)
        for child_node in children_nodes:
            graph.add_edge(node, child_node)
        return node
    return visit

def kaggle(destination, url, combine='cat'):
    def visit(graph, root_dir):
        node = KaggleDownloadNode(os.path.join(root_dir, destination), url, combine)
        graph.add_node(node)
        return node
    return visit

# DSL END -------------------------------------------------------------------

def main():

    stage_1_dataset = interleave(
            destination='UHOx2-wIsRight-multinet-dfrc-n5000-largeGensfen-d9.binpack',
            sources=[
                kaggle(
                    destination='nodes5000pv2_UHO.binpack',
                    url='https://www.kaggle.com/datasets/joostvandevondele/nodes5000pv2-u-uho'
                ),
                kaggle(
                    destination='data_pv-2_diff-100_nodes-5000.binpack',
                    url='https://www.kaggle.com/datasets/joostvandevondele/data-u-pv-2-u-diff-100-u-nodes-5000'
                ),
                kaggle(
                    destination='wrongIsRight_nodes5000pv2.binpack',
                    url='https://www.kaggle.com/datasets/joostvandevondele/wrongisright-u-nodes5000pv2'
                ),
                kaggle(
                    destination='multinet_pv-2_diff-100_nodes-5000.binpack',
                    url='https://www.kaggle.com/datasets/joostvandevondele/multinet-u-pv-2-u-diff-100-u-nodes-5000'
                ),
                kaggle(
                    destination='dfrc_n5000.binpack',
                    url='https://www.kaggle.com/datasets/joostvandevondele/dfrc-u-n5000'
                ),
                kaggle(
                    destination='large_gensfen_multipvdiff_100_d9.binpack',
                    url='https://www.kaggle.com/datasets/joostvandevondele/large-u-gensfen-u-multipvdiff-u-100-u-d9'
                ),
            ]
        )

    graph = DatasetGraph(DATASETS_DIR)
    graph.add_dataset_definition(stage_1_dataset)

    graph.draw()

    additional_usage, additional_download_size = graph.get_additional_required_disk_space_usage()

    print(f'Additional space required: {additional_usage.max}')
    print(f'Additional space used: {additional_usage.current}')
    print(f'Additional download: {additional_download_size}')

    print(f'Proceed? [Y/n]')
    a = input()
    if a == 'Y':
        graph.create(dry=True)

if __name__ == "__main__":
    main()