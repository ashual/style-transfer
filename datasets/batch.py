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

    def get_removable_pads(self):
        removeable_pads = [len([p for p in sentence_pads if p == 0]) for sentence_pads in self.left_padded_masks]
        return min(removeable_pads)

    def clip_redundant_padding(self, pads_to_remove):
        if pads_to_remove > 0:
            self.left_padded_sentences = [s[pads_to_remove:] for s in self.left_padded_sentences]
            self.left_padded_masks = [s[pads_to_remove:] for s in self.left_padded_masks]
            self.right_padded_sentences = [s[:len(s)-pads_to_remove] for s in self.right_padded_sentences]
            self.right_padded_masks = [s[:len(s)-pads_to_remove] for s in self.right_padded_masks]
