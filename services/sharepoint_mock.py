import pandas as pd

class SharePointMock:
    def __init__(self):
        self.path = "data/mock_patients.xlsx"

    def read_patients(self):
        return pd.read_excel(self.path)

    def write_patients(self, df):
        df.to_excel(self.path, index=False)