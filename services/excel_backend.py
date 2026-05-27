import pandas as pd

class ExcelBackend:
    def __init__(self, path):
        self.path = path

    def read_patients(self):
        xls = pd.ExcelFile(self.path)

        sheets = {}
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)

            # normalize column names ON READ (single source of truth)
            df.columns = [
                c.strip().lower().replace(" ", "_")
                for c in df.columns
            ]

            sheets[sheet] = df

        return sheets

    def write_patients(self, sheets_dict):
        with pd.ExcelWriter(self.path, engine="openpyxl", mode="w") as writer:
            for name, df in sheets_dict.items():
                df.to_excel(writer, sheet_name=name, index=False)