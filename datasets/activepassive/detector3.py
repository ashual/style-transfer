from datasets.activepassive.ispassive import Tagger

t = Tagger()
print(t.is_passive('Mistakes were made.'))

print(t.is_passive('I made mistakes'))
print(t.is_passive('adbfadfb'))
