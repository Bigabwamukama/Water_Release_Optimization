from kfold_ml import predict_discharge_forest,predict_discharge_poly,predict_discharge_fnn

class Model():
  def __init__(self,flag = 1):
    self.flag = flag
  def predict(self,water_levels):
    if self.flag == 1:
      return predict_discharge_forest(water_levels=water_levels)
    elif self.flag == 2:
      return predict_discharge_poly(water_levels=water_levels)
    elif self.flag == 3:
      return predict_discharge_fnn(water_levels=water_levels)