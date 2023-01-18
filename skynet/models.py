

class ModelStore:

    def __init__(
        self,
        max_models: int = 2
    ):
        self.max_models = max_models

        self._models = {}

    def get(self, model_name: str):
        if model_name in self._models:
            return self._models[model_name]['pipe']

        if len(self._models) == max_models:
            least_used = list(self._models.keys())[0]
            for model in self._models:
                if self._models[least_used]['generated'] > self._models[model]['generated']:
                    least_used = model

            del self._models[least_used]
            gc.collect()

        pipe = pipeline_for(model_name)

        self._models[model_name] = {
            'pipe': pipe,
            'generated': 0
        }

        return pipe
