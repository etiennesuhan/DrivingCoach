import threading
from FH5 import Listener, Preprocessing, Model


class LivePipeline():
    def __init__(self):
        self.CURRENT_LINE = "data/test.db"
        self.OPTIMAL_LINE = "data/track2_good.db"
        self.queue = []

        # TODO
        #  1) Everything hast to runb in background
        #  2) Preprocessing has to be triggerd after each segment

        listener = Listener(self.CURRENT_LINE)
        threading.Thread(target=listener.run, daemon=True).start()

        preprocessing = Preprocessing(self.CURRENT_LINE, self.OPTIMAL_LINE)
        threading.Thread(target=preprocessing.run, daemon=True).start()
        
        Model(self.queue)