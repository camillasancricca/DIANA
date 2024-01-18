class LOCF:

    def __init__(self):
        self.past_values = [[],[],[]]
        self.added_values = []
        self.predicted_values = []

    def new_value(self,row):
        station = int(row[13])
        for i in range(len(row)):
            if row[i] is None:
                if len(self.past_values[station]) != 0:
                    row[i] = self.past_values[station][i]
                else:
                    row[i] = 0.0
                self.added_values.append(row[i])
        self.past_values[station] = row
        return row



