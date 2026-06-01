import json


class SettingsService:

    def __init__(
        self,
        path="data/clinic_settings.json"
    ):
        self.path = path

    def load_settings(self):

        with open(self.path, "r") as f:
            return json.load(f)

    def save_settings(self, settings):

        with open(self.path, "w") as f:
            json.dump(
                settings,
                f,
                indent=4
            )