import json
import csv


def read_csv_dict(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def write_csv_dict(path, data):
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
