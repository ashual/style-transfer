class Batch:
    def __init__(self):
        self.left_padded_sentences = []
        self.left_padded_masks = []
        self.right_padded_sentences = []
        self.right_padded_masks = []

    def add(self, left_sentence, left_mask, right_sentence, right_mask):
        self.left_padded_sentences.append(left_sentence)
        self.left_padded_masks.append(left_mask)
        self.right_padded_sentences.append(right_sentence)
        self.right_padded_masks.append(right_mask)

    def get_len(self):
        return len(self.left_padded_sentences)
