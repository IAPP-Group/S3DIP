""" Early stopping criterion for Deep Image Prior """

class EarlyStop:
    def __init__(self, size, patience):
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.stop = False
        self.size = size

    def check_stop(self, current, cur_epoch):
        # stop when variance doesn't decrease for consecutive P(patience) times
        best_updated = False  # added
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            should_stop = False
            best_updated = True
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience
        return should_stop, best_updated

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection


def myMetric(x1, x2):
    return ((x1 - x2) ** 2).sum() / x1.size