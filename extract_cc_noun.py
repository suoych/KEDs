import pickle
import os
import multiprocessing
import json
from tqdm import tqdm
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def extract_noun_phrase(text, need_index=False):
    # text = text.lower()

    doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_index:
                return text, [], text
            else:
                other_part = text.replace(head.text, '*').strip()
                return text, other_part # we don't substitude them

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        other_part = text.replace(head.text, '*').strip()
        return head.text, other_part

def extract_target_np(sentence):
    # Parse the sentence with SpaCy
    doc = nlp(sentence)
    
    # Find the root word of the sentence
    root = next((token for token in doc if token.head == token), None)
    if root is None:
        return sentence, sentence  # use the whole sentence if no root word is found
    
    # If the root word is a verb, use its children noun as the root word
    if root.pos_ == 'VERB':
        for child in root.children:
            if child.pos_ == 'NOUN':
                root = child
                break
    
    # Find the noun phrase containing the root word
    target_np = next((np for np in doc.noun_chunks if root in np), None)
    if target_np is None:
        # If no noun phrase containing the root word is found, try to use the children of the root word instead
        children_nouns = [child for child in root.children if child.pos_ == 'NOUN']
        if len(children_nouns) == 1:
            target_np = next((np for np in doc.noun_chunks if children_nouns[0] in np), None)
    
    if target_np is None:
        return sentence, sentence  # use the whole sentence if still no target noun phrase is found
    
    other_part = sentence.replace(target_np.text, '*').strip()
    return target_np.text, other_part

cap_dict = {}
with open("cc3m_have_good.pkl","rb") as f_g:
    img_cap_list_good = pickle.load(f_g)

with open("cc3m_have.pkl","rb") as f:
    img_cap_list = pickle.load(f)

for i in range(len(img_cap_list_good)):
    cap_dict[img_cap_list_good[i]['filename']] = img_cap_list_good[i]['text']

for i in range(len(img_cap_list)):
    cap_dict[img_cap_list[i]['filename']] = img_cap_list[i]['text']


# Create a multiprocessing pool
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

# Use the pool to extract subjects and words for each caption
#results = pool.map(extract_noun_phrase, cap_dict.values())
results = list(tqdm(pool.imap(extract_noun_phrase, cap_dict.values()), total=len(cap_dict)))

# Close the pool to release resources
pool.close()
pool.join()

# Initialize empty dictionaries for subjects and words
subject_dict = {}
words_dict = {}

for key, result in zip(cap_dict.keys(), results):
    subject, words_beside_subject = result
    subject_dict[key] = subject
    words_dict[key] = words_beside_subject

with open("cc_subject.json", 'w') as subject_file:
    json.dump(subject_dict, subject_file)

# Dump words_dict to a JSON file
with open("cc_other.json", 'w') as words_file:
    json.dump(words_dict, words_file)