import os
import math
import sys


def process_text(text):
    proportions = [0.0] * 26
    text = text.lower()

    letters_only = [c for c in text if 'a' <= c <= 'z']
    total_letters = len(letters_only)

    if total_letters == 0:
        return proportions

    for c in letters_only:
        index = ord(c) - ord('a')
        proportions[index] += 1.0

    for i in range(26):
        proportions[i] /= total_letters

    return proportions


def normalize_euclidean(vector):
    norm = math.sqrt(sum(v ** 2 for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class Perceptron:
    def __init__(self, language):
        self.language = language
        self.weights = [0.0] * 26
        self.theta = 0.0
        self.alpha = 0.1

    def predict_discrete(self, x):
        net = sum(w * xi for w, xi in zip(self.weights, x)) - self.theta
        return 1 if net >= 0 else 0

    def train(self, x, d):
        y = self.predict_discrete(x)
        error = d - y

        if error != 0:
            for i in range(26):
                self.weights[i] += self.alpha * error * x[i]
            self.theta -= self.alpha * error
            return True
        return False

    def get_continuous_score(self, x_norm):
        w_norm = normalize_euclidean(self.weights)
        return sum(w * xi for w, xi in zip(w_norm, x_norm))


def load_data(base_folder):
    training_data = []
    languages = []

    target_languages = ['angielski', 'czeski', 'polski']

    for lang_folder in target_languages:
        lang_path = os.path.join(base_folder, lang_folder)
        if os.path.isdir(lang_path):
            languages.append(lang_folder)
            for file_name in os.listdir(lang_path):
                if file_name.endswith('.txt'):
                    with open(os.path.join(lang_path, file_name), 'r', encoding='utf-8') as file:
                        text = file.read()
                        vector = process_text(text)
                        training_data.append((vector, lang_folder))
        else:
            print(f"Ostrzeżenie: Nie znaleziono folderu '{lang_path}'.")

    return training_data, languages


def train_network(training_data, languages):
    perceptrons = [Perceptron(lang) for lang in languages]
    epochs = 1000

    for epoch in range(epochs):
        has_errors = False
        for x, true_lang in training_data:
            for p in perceptrons:
                d = 1 if p.language == true_lang else 0
                if p.train(x, d):
                    has_errors = True
        if not has_errors:
            print(f"Sieć nauczona z sukcesem w {epoch + 1} epokach.")
            break

    return perceptrons


def classify_text(text, perceptrons):
    x = process_text(text)
    x_norm = normalize_euclidean(x)

    best_score = -float('inf')
    best_lang = "Nieznany"

    for p in perceptrons:
        score = p.get_continuous_score(x_norm)
        print(f"[{p.language}] - Pewność sieci: {score:.4f}")
        if score > best_score:
            best_score = score
            best_lang = p.language

    return best_lang


if __name__ == "__main__":
    folder_danych = os.path.dirname(os.path.abspath(__file__))

    dane_treningowe, lista_jezykow = load_data(folder_danych)

    if not lista_jezykow:
        print("Błąd Krytyczny: Nie znaleziono żadnych danych treningowych!")
        sys.exit(1)

    siec = train_network(dane_treningowe, lista_jezykow)

    print("\n=== SYSTEM GOTOWY ===")
    print(f"Rozpoznawane języki: {', '.join(lista_jezykow)}")

    while True:
        try:
            print("\n------------------------------------------------")
            testowy_tekst = input("Wklej tekst do rozpoznania (lub wpisz 'exit' aby wyjść):\n> ")

            if testowy_tekst.lower() == 'exit':
                print("Zamykanie programu...")
                break

            if len(testowy_tekst.strip()) < 10:
                print("Tekst jest zbyt krótki do analizy. Podaj więcej treści.")
                continue

            zwyciezca = classify_text(testowy_tekst, siec)
            print(f"\n=> WERDYKT: Tekst napisano w języku: {zwyciezca.upper()}")
        except (EOFError, KeyboardInterrupt):
            break