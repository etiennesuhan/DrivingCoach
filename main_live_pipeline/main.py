from listener import Listener
from model import Model

CURRENT_LINE = "data/test.db"

# print(f"Lade Modell: {Model().MODEL_NAME}")
# Model().warmup_model()
Listener(CURRENT_LINE).run()