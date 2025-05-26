from pathlib import Path
import csv
import json

base_path = Path(__file__).parent

def csv_column_to_list(csv_path='ONET_data\ONET_DWA.csv', column_name='DWA Title'):
    csv_path = base_path / csv_path
    extracted_data = []

    with open(csv_path, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            extracted_data.append(row[column_name])
    
    return extracted_data


def json_to_list(json_path='university_data\sample_data.json'):
    json_path = base_path / json_path
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)
