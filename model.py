from catboost import CatBoostClassifier

from pick_regno import pick_regno


class PickRegnoModel:
    def __init__(self, model_path: str) -> None:
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def predict(self, data: dict):
        return pick_regno(
            data["camera_regno"],
            data["nn_regno"],
            data["camera_score"],
            data["nn_score"],
            data["nn_sym_scores"],
            data["nn_len_scores"],
            data["camera_type"],
            data["camera_class"],
            data["time_check"],
            data["direction"],
            self.model
        )
