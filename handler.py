import os

from model import PickRegnoModel


class Handler:
    def __init__(self) -> None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "weights", "micromodel.cbm")

        self.model = PickRegnoModel(model_path)

    async def pipe(self, data: dict):
        try:
            result = self.model.predict(data)
            return {"result": result.tolist()}
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")
