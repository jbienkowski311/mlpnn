import csv
from random import shuffle


class InputHandler(object):
    @staticmethod
    def handle(input_file, shuffle_data=False):
        instance = InputHandler()

        data = instance.import_file(input_file.path)

        if shuffle_data:
            shuffle(data)

        return data

    def import_file(self, input_file):
        output_data = list()

        with open(input_file) as csv_file:
            reader = csv.reader(csv_file)

            for input_data in reader:
                output_data.append(input_data)

        return output_data
