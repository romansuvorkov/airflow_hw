import os
import dill
import pandas as pd
import json
from datetime import datetime
path = os.environ.get('PROJECT_PATH', '.')


def sort_list(input, number):
    def last_chars(string):
        splitted_string = string.split('.')
        return (splitted_string[0][-number:])
    output = sorted(input, key=last_chars, reverse=True)
    return output


def get_prediction(names_arr, model):
    output = pd.DataFrame(columns=['car_id', 'price_category'])
    for file_name in names_arr:
        with open(f'{path}/data/test/{file_name}', 'rb') as test_file:
            data = json.load(test_file)
        df = pd.json_normalize(data)
        y = model.predict(df)
        new_row = pd.DataFrame({'car_id': df['id'], 'price_category': y[0]})
        output = pd.concat([output, new_row], ignore_index=True)
    return output


def predict():
    models_arr = os.listdir(f'{path}/data/models')
    sorted_models_arr = sort_list(models_arr, 12)
    with open(f'{path}/data/models/{sorted_models_arr[0]}', 'rb') as file:
        model = dill.load(file)
    file_names_arr = os.listdir(f'{path}/data/test')
    output = get_prediction(file_names_arr, model)
    output.to_csv(f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
