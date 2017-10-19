class DataPipeline:

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def process(self, data):
        for step in self._pipeline:
            if data is None:
                return None
            data = step(data)
        return data

    def __call__(self, s):
        return self.process(s)
