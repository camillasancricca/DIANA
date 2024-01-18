from river.anomaly import HalfSpaceTrees
from river.compose import Pipeline
from river.preprocessing import MinMaxScaler

class HST:
    def __init__(self):
        self.model = Pipeline(
            MinMaxScaler(),
            HalfSpaceTrees(
                n_trees=100,
                height=10,
                seed=42)
        )

    def add_sample(self,row):
        row = row.select_dtypes(exclude=['object', 'boolean']).to_dict('records')[0]
        self.model.learn_one(row)
        score = self.model.score_one(row)
        if score >= 0.7:
            return True
        else:
            return False