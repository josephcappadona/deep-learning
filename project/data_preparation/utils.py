import json


def get_filename(filepath):
    return filepath.split('/')[-1]

def get_parent_folder(filepath):
    return filepath.split('/')[-2]

def remove_extension(filename):
    return filename[:filename.rfind('.')]

def read_json(filepath):
    with open(filepath, 'rt') as f:
        return json.load(f)

def write_json(filepath, data):
    with open(filepath, 'w+t') as f:
        return json.dump(data, f)
