import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
import string
import numpy as np
import spacy
nlp = spacy.load('fr_core_news_sm')


from spellchecker import SpellChecker
spacy.cli.download("fr_core_news_sm")
nlp = spacy.load("fr_core_news_sm")
import language_tool_python
tool = language_tool_python.LanguageTool('fr')
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def evaluer_orthographe_syntaxe(texte):
    erreurs_language_tool = tool.check(texte)
    erreurs_orthographe = sum(1 for erreur in erreurs_language_tool if 'ORTHOGRAPH' in erreur.ruleId)
    erreurs_grammaire = sum(1 for erreur in erreurs_language_tool if 'GRAMMAR' in erreur.ruleId)
    doc = nlp(texte)
    erreurs_syntaxe = sum(1 for token in doc if token.dep_ == "nsubj" and token.head.pos_ != 'VERB')
    seuil_minimal_mots = 5
    nombre_mots = max(len(texte.split()), seuil_minimal_mots)
    poids_orthographe = 1.0  
    poids_grammaire = 1.5
    note_globale = max(1 - ((erreurs_orthographe * poids_orthographe + erreurs_grammaire * poids_grammaire + erreurs_syntaxe) / nombre_mots), 0)

    return note_globale

def preprocess_text(texte, remove_stopwords=True):
    if not isinstance(texte, str):
        raise ValueError("Le texte doit être une chaîne de caractères.")

    mots = word_tokenize(texte, language='french')
    mots_low = [mot.lower() for mot in mots if mot.isalpha()] 
    return mots_low

def diversite_lexicale_complexite(texte, remove_stopwords=True):
    phrases = sent_tokenize(texte, language='french')
    mots_low = preprocess_text(texte, remove_stopwords)
    
    if not mots_low or not phrases:
        return float(0) 
    
    nb_mots = len(mots_low)
    nb_phrases = len(phrases)
    longueur_moyenne_mot = sum(len(mot) for mot in mots_low) / nb_mots
    longueur_moyenne_phrase = sum(len(phrase.split()) for phrase in phrases) / nb_phrases
    lexical_diversity = len(set(mots_low)) / nb_mots
    complexite = lexical_diversity * (1 + (longueur_moyenne_mot / 5)) * (1 + (longueur_moyenne_phrase / 10))
    
    return complexite

def sentence_length(sentence):
    return len(sentence.split())

def average_word_length(sentence):
    words = sentence.split()
    return np.mean([len(word) for word in words]) if words else 0

def type_token_ratio(sentence):
    words = sentence.split()
    return len(set(words)) / len(words) if words else 0

def complexite_texte(texte):
    doc = nlp(texte)

    # Syntactic measurements
    nb_phrases = len(list(doc.sents))
    profondeur_moyenne = sum(len(list(phrase.root.subtree)) for phrase in doc.sents) / nb_phrases if nb_phrases > 0 else 0

    # Grammatical measures
    temps_verbaux = {mot.tag_: 0 for mot in doc if mot.tag_ and "VERB" in mot.tag_}
    for mot in doc:
        if mot.tag_ and "VERB" in mot.tag_:
            temps_verbaux[mot.tag_] += 1
    diversite_temps_verbaux = len(temps_verbaux)

    complexite = profondeur_moyenne + diversite_temps_verbaux

    return complexite

def pos_tag_distribution(sentence):
    if not isinstance(sentence, str):
        raise ValueError("L'entrée doit être une chaîne de caractères.")

    doc = nlp(sentence)
    pos_counts = {pos: 0 for pos in [token.pos_ for token in doc]}  

    for token in doc:
        pos = token.pos_
        pos_counts[pos] += 1

    total_mots = len(doc)
    if total_mots > 0:
        pos_counts_normalized = {pos: count / total_mots for pos, count in pos_counts.items()}
        return pos_counts_normalized

    return pos_counts



def augmenter_dataframe(df):
    # Évaluation des caractéristiques linguistiques pour chaque phrase
    df['note_orthographe'] = df['sentence'].apply(evaluer_orthographe_syntaxe)
    df['lexical_complexite'] = df['sentence'].apply(diversite_lexicale_complexite)
    df['char_length'] = df['sentence'].apply(len)
    df['word_length'] = df['sentence'].apply(lambda x: len(x.split()))
    df['type_token_ratio'] = df['sentence'].apply(type_token_ratio)
    df['sentence_length'] = df['sentence'].apply(sentence_length)
    df['avg_word_length'] = df['sentence'].apply(average_word_length)
    df['complexite_texte'] = df['sentence'].apply(complexite_texte)
    df['pos_tags'] = df['sentence'].apply(pos_tag_distribution)

    # Traitement des colonnes POS tags
    unique_pos_tags = set()
    for pos_tags_dict in df['pos_tags']:
        unique_pos_tags.update(pos_tags_dict.keys())

    # Initialize columns for each POS tag with default value 0
    for tag in ['PUNCT', 'ADV', 'CCONJ', 'X', 'AUX', 'DET', 'PRON', 'NUM', 'NOUN', 'INTJ', 'ADP', 'ADJ', 'VERB', 'PROPN', 'SCONJ']:
        df[tag] = 0

    # Populate the columns with counts
    for index, row in df.iterrows():
        for tag, count in row['pos_tags'].items():
            if tag in df.columns:
                df.at[index, tag] = count

    df = df.drop(['pos_tags'], axis=1)

    numerical_features = ['lexical_complexite', 'note_orthographe', 'char_length', 'word_length', 'type_token_ratio', 'sentence_length', 'avg_word_length', 'complexite_texte']

    # scaler MinMax
    scaler = MinMaxScaler()

    df[numerical_features] = scaler.fit_transform(df[numerical_features])



    return df
