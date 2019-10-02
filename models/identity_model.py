from models.base_model import Model

class IdentityModel(Model):
    def __init__(self):
        self.name = "identity_model"

    def reformulate(self,query):
        return query