import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import logging
import os
from collections import Counter

# Ignorowanie ostrzeżeń Pythona
warnings.filterwarnings("ignore")

# Ignorowanie logów ostrzeżeń
logging.basicConfig(level=logging.ERROR)

# Ustawienie zmiennych środowiskowych
os.environ['PYTHONWARNINGS'] = 'ignore'


# Inicjalizacja zmiennych globalnych
df = None
df_imputed = None
df_normalized = None
df_standardized = None
df_discretized = None
current_processed_data = None


def custom_knn_imputer(data, k=3, weights="uniform", metric="euclidean"):
    """
    Custom implementation of KNN imputer mimicking scikit-learn's implementation.

    Args:
        data (pd.DataFrame): Pandas DataFrame with missing values.
        k (int): Number of neighbors to consider for imputing.
        weights (str): Weighting function ("uniform" or "distance").
        metric (str): Distance metric to use ("euclidean" is supported).

    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    if metric != "euclidean":
        raise ValueError("Only 'euclidean' metric is supported in this implementation.")
    if weights not in {"uniform", "distance"}:
        raise ValueError("Weights must be either 'uniform' or 'distance'.")

    # Convert to numpy array
    data_array = data.to_numpy()
    missing_mask = np.isnan(data_array)

    imputed_values = data_array.copy()

    for row_idx, row in enumerate(data_array):
        for col_idx, value in enumerate(row):
            if missing_mask[row_idx, col_idx]:  # Missing value detected
                # Calculate distances to other rows (ignoring the current row)
                distances = []
                valid_neighbors = []

                for neighbor_idx, neighbor_row in enumerate(data_array):
                    if neighbor_idx != row_idx and not np.isnan(neighbor_row[col_idx]):
                        # Compute distance on non-NaN shared features
                        non_nan_indices = ~missing_mask[row_idx] & ~missing_mask[neighbor_idx]
                        if np.any(non_nan_indices):
                            dist = np.sqrt(np.sum((row[non_nan_indices] - neighbor_row[non_nan_indices]) ** 2))
                            distances.append((dist, neighbor_row[col_idx]))

                # Sort neighbors by distance
                distances.sort(key=lambda x: x[0])
                valid_neighbors = distances[:k]

                if valid_neighbors:
                    if weights == "uniform":
                        # Take mean of the k nearest neighbors
                        imputed_value = np.mean([val for _, val in valid_neighbors])
                    elif weights == "distance":
                        # Weighted mean of the k nearest neighbors (1/distance weighting)
                        weighted_values = [val / dist if dist > 0 else val for dist, val in valid_neighbors]
                        weights_sum = np.sum([1 / dist if dist > 0 else 1 for dist, _ in valid_neighbors])
                        imputed_value = np.sum(weighted_values) / weights_sum

                    # Assign imputed value to the missing cell
                    imputed_values[row_idx, col_idx] = imputed_value

    return pd.DataFrame(imputed_values, columns=data.columns)


class CustomRandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, random_state=None):
        # Inicjalizacja parametrów lasu losowego
        self.n_estimators = n_estimators  # Liczba drzew w lesie
        self.max_depth = max_depth  # Maksymalna głębokość każdego drzewa
        self.min_samples_split = min_samples_split  # Minimalna liczba próbek do podziału węzła
        self.random_state = random_state  # Ziarno losowości dla powtarzalności wyników
        self.trees = []  # Lista na przechowywanie drzew

    class DecisionTree:
        def __init__(self, max_depth=10, min_samples_split=2):
            # Inicjalizacja parametrów pojedynczego drzewa
            self.max_depth = max_depth  # Maksymalna głębokość drzewa
            self.min_samples_split = min_samples_split  # Minimalna liczba próbek do podziału
            self.tree = None  # Struktura drzewa

        def gini(self, y):
            # Obliczanie współczynnika Giniego dla danego zbioru etykiet
            _, counts = np.unique(y, return_counts=True)  # Zliczanie wystąpień każdej klasy
            probabilities = counts / len(y)  # Obliczanie prawdopodobieństw
            return 1 - np.sum(probabilities ** 2)  # Wzór na współczynnik Giniego

        def split_data(self, X, y, feature_idx, threshold):
            # Podział danych na podstawie wartości progowej dla wybranej cechy
            left_mask = X[:, feature_idx] <= threshold  # Maska dla lewej strony podziału
            # Zwraca dane podzielone na lewą i prawą stronę
            return (X[left_mask], y[left_mask], X[~left_mask], y[~left_mask])

        def find_best_split(self, X, y):
            # Znajdowanie najlepszego podziału danych
            best_gini = float('inf')  # Inicjalizacja najlepszego współczynnika Giniego
            best_split = None  # Inicjalizacja najlepszego podziału

            # Przeszukiwanie wszystkich cech i możliwych wartości progowych
            for feature_idx in range(X.shape[1]):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    # Podział danych dla danej wartości progowej
                    X_left, y_left, X_right, y_right = self.split_data(X, y, feature_idx, threshold)
                    if len(y_left) == 0 or len(y_right) == 0:
                        continue  # Pomijanie podziałów, które dają puste zbiory

                    # Obliczanie ważonego współczynnika Giniego
                    gini = (len(y_left) * self.gini(y_left) + len(y_right) * self.gini(y_right)) / len(y)
                    # Aktualizacja najlepszego podziału
                    if gini < best_gini:
                        best_gini = gini
                        best_split = (feature_idx, threshold)

            return best_split

        def build_tree(self, X, y, depth=0):
            # Rekurencyjne budowanie drzewa decyzyjnego
            n_samples = len(y)

            # Warunki stopu rekurencji
            if (depth >= self.max_depth or
                    n_samples < self.min_samples_split or
                    len(np.unique(y)) == 1):
                return Counter(y).most_common(1)[0][0]  # Zwraca najczęstszą klasę

            # Znajdowanie najlepszego podziału
            best_split = self.find_best_split(X, y)
            if best_split is None:
                return Counter(y).most_common(1)[0][0]

            # Tworzenie węzła drzewa i rekurencyjne budowanie poddrzew
            feature_idx, threshold = best_split
            X_left, y_left, X_right, y_right = self.split_data(X, y, feature_idx, threshold)

            return {
                'feature_idx': feature_idx,
                'threshold': threshold,
                'left': self.build_tree(X_left, y_left, depth + 1),
                'right': self.build_tree(X_right, y_right, depth + 1)
            }

        def fit(self, X, y):
            # Trenowanie drzewa
            self.tree = self.build_tree(X, y)

        def predict_single(self, x, tree):
            # Predykcja dla pojedynczej próbki
            if not isinstance(tree, dict):
                return tree  # Zwraca klasę jeśli to liść
            # Rekurencyjne przechodzenie przez drzewo
            if x[tree['feature_idx']] <= tree['threshold']:
                return self.predict_single(x, tree['left'])
            return self.predict_single(x, tree['right'])

        def predict(self, X):
            # Predykcja dla wielu próbek
            return np.array([self.predict_single(x, self.tree) for x in X])

    def fit(self, X, y):
        # Trenowanie lasu losowego
        if self.random_state is not None:
            np.random.seed(self.random_state)  # Ustawienie ziarna losowości

        n_samples = X.shape[0]
        self.trees = []

        # Tworzenie i trenowanie poszczególnych drzew
        for _ in range(self.n_estimators):
            # Bootstrap sampling - losowanie próbek z powtórzeniami
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Tworzenie i trenowanie pojedynczego drzewa
            tree = self.DecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        # Predykcja całego lasu losowego
        # Zbieranie predykcji ze wszystkich drzew
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Głosowanie większościowe dla każdej próbki
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0]
                         for i in range(predictions.shape[1])])

# Funkcja do ładowania pliku CSV
def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Sukces", "Plik CSV załadowany pomyślnie!")
            show_original_csv()
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się załadować pliku: {e}")
    else:
        messagebox.showwarning("Brak pliku", "Nie wybrano żadnego pliku.")


# Funkcja do eksportu danych do CSV
# noinspection PyUnresolvedReferences
def export_to_csv():
    global current_processed_data
    if current_processed_data is None:
        messagebox.showwarning("Błąd", "Brak danych do eksportu!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Eksportuj dane przetworzone do CSV"
    )

    if file_path:
        try:
            current_processed_data.to_csv(file_path, index=False, encoding='utf-8')
            messagebox.showinfo("Sukces", f"Dane zostały wyeksportowane do:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wyeksportować pliku: {e}")


# Funkcja do wyświetlania oryginalnego CSV w tabeli
# noinspection PyUnresolvedReferences,PyArgumentList
def show_original_csv():
    if df is None:
        return

    for widget in original_data_frame.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(original_data_frame, columns=list(df.columns), show='headings', height=10)
    style.configure("Treeview", font=("Consolas", 10), rowheight=25)
    style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    scrollbar = ttk.Scrollbar(original_data_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)


# Funkcja do wyświetlania przetworzonych danych
# noinspection PyUnresolvedReferences,PyArgumentList
def show_csv(data: object) -> object:
    global current_processed_data
    current_processed_data = data

    # Sprawdzanie, czy dane zostały załadowane
    if df is None:
        return

    # Czyszczenie poprzednich widoków
    for widget in processed_data_frame.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(processed_data_frame, columns=list(data.columns), show='headings', height=10)
    style.configure("Treeview", font=("Consolas", 10), rowheight=25)
    style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))

    # Tworzenie nagłówków tabeli
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    # Porównanie wartości w kolumnie 'porzucenie'
    for index, row in data.iterrows():
        original_row = df.iloc[index]
        row_values = list(row)

        # Jeśli wartość w kolumnie 'porzucenie' zmieniła się, podkreśl ją i dodaj informację
        for col_index, col in enumerate(data.columns):
            if col == 'porzucenie' and row['porzucenie'] != original_row['porzucenie']:
                # Dodaj strzałkę wskazującą zmianę
                row_values[col_index] = f"{original_row['porzucenie']} → {row_values[col_index]} (ZMIANA)"
                # Możesz również dodać specjalny tag do wyróżnienia
                tree.tag_configure('changed', background='#FF6347')  # Kolor czerwonawy

        # Wstawianie wartości do drzewa
        if row['porzucenie'] != original_row['porzucenie']:
            tree.insert("", "end", values=row_values, tags=('changed',))
        else:
            tree.insert("", "end", values=row_values)

    scrollbar = ttk.Scrollbar(processed_data_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    # Wyświetlanie tabeli
    tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)


# noinspection PyTypeChecker,PyUnresolvedReferences
def show_change_statistics():
    global df, current_processed_data

    if df is None or current_processed_data is None:
        messagebox.showwarning("Błąd", "Najpierw przetwórz dane!")
        return

    # Zliczenie zmian w decyzji
    changes = 0
    total_rows = len(df)

    for index, row in current_processed_data.iterrows():
        if row['porzucenie'] != df.iloc[index]['porzucenie']:
            changes += 1

    # Obliczenie skuteczności zmiany
    change_percentage = (changes / total_rows) * 100

    # Przygotowanie statystyk
    stats_text = f"""
Statystyki przetwarzania danych:
------------------------------
Całkowita liczba wierszy: {total_rows}
Liczba zmienionych wierszy: {changes}
Procent zmienionych wierszy: {change_percentage:.2f}%

Interpretacja:
- Algorytm zmienił decyzję w {changes} wierszach
- {change_percentage:.2f}% danych zostało zmodyfikowanych
"""

    # Utworzenie nowego okna ze statystykami
    stats_window = tk.Toplevel(root)
    stats_window.title("Statystyki zmian")
    stats_window.geometry("500x300")
    stats_window.configure(bg=BACKGROUND_COLOR)

    stats_label = ttk.Label(
        stats_window,
        text=stats_text,
        font=("Consolas", 12),
        background=BACKGROUND_COLOR,
        foreground=TEXT_COLOR,
        justify="left"
    )
    stats_label.pack(padx=20, pady=20, expand=True, fill="both")

# Funkcje przetwarzania



def apply_knn_imputer():
    global df, df_imputed
    if df is None:
        messagebox.showwarning("Błąd", "Załaduj najpierw plik CSV!")
        return

    knn_method = knn_method_var.get()  # Get selected method
    try:
        # Separate features and target
        features = df.drop('porzucenie', axis=1)
        target = df['porzucenie']

        if knn_method == "Scikit-learn":
            knn_imputer = KNNImputer(n_neighbors=4)
            imputed_features = knn_imputer.fit_transform(features)
        elif knn_method == "Custom":
            imputed_features = custom_knn_imputer(features, k=4)
        else:
            raise ValueError("Nieznana metoda imputacji KNN.")

        # Reconstruct DataFrame with imputed features and original target
        df_imputed = pd.DataFrame(imputed_features, columns=features.columns)
        df_imputed['porzucenie'] = target

        messagebox.showinfo("Sukces", f"Imputacja {knn_method} zakończona!")
        show_csv(df_imputed)
    except Exception as e:
        messagebox.showerror("Błąd", f"Imputacja zakończona niepowodzeniem: {e}")

# noinspection PyUnresolvedReferences



# noinspection PyUnresolvedReferences
def apply_normalization():
    global df_imputed, df_normalized
    if df_imputed is None:
        messagebox.showwarning("Błąd", "Najpierw wykonaj imputację kNN!")
        return
    minmax_scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        minmax_scaler.fit_transform(df_imputed.drop('porzucenie', axis=1)),
        columns=df_imputed.columns[:-1]
    )
    df_normalized['porzucenie'] = df_imputed['porzucenie']
    messagebox.showinfo("Sukces", "Normalizacja zakończona!")
    show_csv(df_normalized)


# noinspection PyUnresolvedReferences
def apply_standardization():
    global df_normalized, df_standardized
    if df_normalized is None:
        messagebox.showwarning("Błąd", "Najpierw wykonaj normalizację!")
        return
    standard_scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        standard_scaler.fit_transform(df_normalized.drop('porzucenie', axis=1)),
        columns=df_normalized.columns[:-1]
    )
    df_standardized['porzucenie'] = df_normalized['porzucenie']
    messagebox.showinfo("Sukces", "Standaryzacja zakończona!")
    show_csv(df_standardized)


# noinspection PyUnresolvedReferences
def apply_discretization():
    global df_standardized, df_discretized
    if df_standardized is None:
        messagebox.showwarning("Błąd", "Najpierw wykonaj standaryzację!")
        return
    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    df_discretized = pd.DataFrame(
        discretizer.fit_transform(df_standardized.drop('porzucenie', axis=1)),
        columns=df_standardized.columns[:-1]
    )
    df_discretized['porzucenie'] = df_standardized['porzucenie']
    messagebox.showinfo("Sukces", "Dyskretyzacja zakończona!")
    show_csv(df_discretized)


# noinspection PyUnresolvedReferences
def apply_random_forest():
    # Funkcja aplikująca las losowy do danych
    global df_discretized

    # Sprawdzenie czy dane są zdyskretyzowane
    if df_discretized is None:
        messagebox.showwarning("Błąd", "Najpierw wykonaj dyskretyzację!")
        return

    # Przygotowanie danych do klasyfikacji
    X = df_discretized.drop('porzucenie', axis=1).values  # Cechy
    y = df_discretized['porzucenie'].values              # Etykiety

    # Podział na zbiór treningowy i testowy (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pobranie wybranej metody (Scikit-learn lub Custom)
    rf_method = rf_method_var.get()

    try:
        # Wybór i inicjalizacja odpowiedniego klasyfikatora
        if rf_method == "Scikit-learn":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Custom
            rf = CustomRandomForest(n_estimators=100, random_state=42)

        # Trenowanie modelu i wykonanie predykcji
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Wyświetlenie wyników
        messagebox.showinfo("Sukces",
                            f"Dokładność klasyfikacji Random Forest ({rf_method}): {accuracy * 100:.2f}%")

        # Aktualizacja decyzji w danych
        df_discretized['porzucenie'] = rf.predict(X)
        show_csv(df_discretized)

    except Exception as e:
        # Obsługa błędów
        messagebox.showerror("Błąd", f"Wystąpił problem podczas klasyfikacji: {e}")


# noinspection PyUnresolvedReferences
def run_pipeline():
    global df, df_discretized
    if df is None:
        messagebox.showwarning("Błąd", "Załaduj najpierw plik CSV!")
        return

    try:
        # Imputacja kNN
        knn_imputer = KNNImputer(n_neighbors=3)
        df_imputed1 = pd.DataFrame(
            knn_imputer.fit_transform(df.drop('porzucenie', axis=1)),
            columns=df.columns[:-1]
        )
        df_imputed1['porzucenie'] = df['porzucenie']

        # Normalizacja
        minmax_scaler = MinMaxScaler()
        df_normalized1 = pd.DataFrame(
            minmax_scaler.fit_transform(df_imputed1.drop('porzucenie', axis=1)),
            columns=df_imputed1.columns[:-1]
        )
        df_normalized1['porzucenie'] = df_imputed1['porzucenie']

        # Standaryzacja
        standard_scaler = StandardScaler()
        df_standardized1 = pd.DataFrame(
            standard_scaler.fit_transform(df_normalized1.drop('porzucenie', axis=1)),
            columns=df_normalized1.columns[:-1]
        )
        df_standardized1['porzucenie'] = df_normalized1['porzucenie']

        # Dyskretyzacja
        discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
        df_discretized = pd.DataFrame(
            discretizer.fit_transform(df_standardized1.drop('porzucenie', axis=1)),
            columns=df_standardized1.columns[:-1])

        df_discretized['porzucenie'] = df_standardized1['porzucenie']

        # Random Forest na końcu potoku
        apply_random_forest()

        messagebox.showinfo("Sukces", "Potok przetwarzania danych zakończony!")

    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił problem podczas przetwarzania danych: {e}")

# Funkcja do wyświetlenia legendy


def show_legend():
    legend_text = """
Legenda (opis kolumn)
Wiek: Wiek użytkownika w latach (18-65).
Płeć: Płeć użytkownika (0 - kobieta, 1 - mężczyzna).
Lokalizacja: Lokalizacja użytkownika (1 - region A, 2 - region B, 3 - region C).
Rodzaj uslugi: Typ usługi używanej przez użytkownika (1 - podstawowa, 2 - premium, 3 - VIP).
Czas subskrypcji: Czas trwania subskrypcji w miesiącach.
Czas na platformie: Łączny czas spędzony na platformie w godzinach.
Liczba reklamacji: Liczba reklamacji zgłoszonych przez użytkownika.
Srednia transakcji: Średnia liczba transakcji miesięcznych.
Srednie wydatki: Średnie miesięczne wydatki użytkownika w jednostkach walutowych.
Zalegle platnosci: Liczba zaległych płatności użytkownika.
Rabaty: Wartość rabatów przyznanych użytkownikowi (%).
Porzucenie: Flaga porzucenia (0 - użytkownik pozostał, 1 - użytkownik zrezygnował z subskrypcji).
"""
    legend_window = tk.Toplevel(root)
    legend_window.title("Legenda CSV")
    legend_window.geometry("700x400")

    legend_window.configure(bg="#2c3e50")

    legend_label = ttk.Label(
        legend_window,
        text=legend_text,
        justify="left",
        wraplength=650,
        font=("Segoe UI", 10),
        background="#2c3e50",
        foreground="#ecf0f1"
    )
    legend_label.pack(padx=15, pady=15)


# Tworzenie głównego okna aplikacji
root = tk.Tk()
root.title("Zaawansowane Przetwarzanie Danych")
root.geometry("1600x900")
root.configure(bg="#2c3e50")

# Ulepszona stylizacja
style = ttk.Style()
style.theme_use("clam")

# Własne kolory
PRIMARY_COLOR = "#3498db"  # Niebieski
SECONDARY_COLOR = "#2ecc71"  # Zielony
BACKGROUND_COLOR = "#2c3e50"  # Ciemny granat
TEXT_COLOR = "#ecf0f1"  # Jasny, czysty biały
BUTTON_COLOR = "#34495e"  # Ciemniejszy odcień niebieskiego

# Niestandardowe style dla elementów
style.configure("TButton",
                font=("Segoe UI", 12, "bold"),
                padding=10,
                background=BUTTON_COLOR,
                foreground=TEXT_COLOR
                )

style.map("TButton",
          background=[('active', PRIMARY_COLOR)],
          foreground=[('active', 'white')]
          )

style.configure("TLabel",
                font=("Segoe UI", 12),
                background=BACKGROUND_COLOR,
                foreground=TEXT_COLOR
                )

style.configure("TFrame",
                background=BACKGROUND_COLOR
                )

style.configure("TLabelframe",
                background=BACKGROUND_COLOR,
                foreground=TEXT_COLOR
                )

style.configure("TLabelframe.Label",
                font=("Segoe UI", 14, "bold"),
                background=BACKGROUND_COLOR,
                foreground=TEXT_COLOR
                )

style.configure("Treeview",
                background="#34495e",
                foreground=TEXT_COLOR,
                fieldbackground="#34495e",
                font=("Consolas", 10)
                )

style.configure("Treeview.Heading",
                font=("Segoe UI", 12, "bold"),
                background=PRIMARY_COLOR,
                foreground=TEXT_COLOR
                )

# Sekcja przycisków z bardziej responsywnym układem
button_frame = ttk.Frame(root)
button_frame.pack(side="top", fill="x", pady=15, padx=15)
# Add a dropdown menu to choose the KNN method
knn_method_var = tk.StringVar(value="Scikit-learn")
knn_method_dropdown = ttk.Combobox(
    button_frame,
    textvariable=knn_method_var,
    values=["Scikit-learn", "Custom"],
    state="readonly"
)
knn_method_dropdown.pack(side="left", padx=5)
rf_method_var = tk.StringVar(value="Scikit-learn")
rf_method_dropdown = ttk.Combobox(
    button_frame,
    textvariable=rf_method_var,
    values=["Scikit-learn", "Custom"],
    state="readonly"
)
rf_method_dropdown.pack(side="left", padx=5)


# Lista przycisków z bardziej dynamicznym stylem
buttons_config = [
    ("Załaduj CSV", load_csv, PRIMARY_COLOR),
    ("Imputacja kNN", apply_knn_imputer, SECONDARY_COLOR),
    ("Normalizacja", apply_normalization, "#e74c3c"),  # Czerwony
    ("Standaryzacja", apply_standardization, "#f39c12"),  # Pomarańczowy
    ("Dyskretyzacja", apply_discretization, "#9b59b6"),  # Fioletowy
    ("Uruchom potok", run_pipeline, "#1abc9c"),  # Morski
    ("Pokaż legendę", show_legend, "#95a5a6"),  # Szary
    ("Eksportuj CSV", export_to_csv, "#27ae60"),  # Zielony
    ("Statystyki zmian", show_change_statistics, "#27ae60")
]


for (text, command, color) in buttons_config:
    btn = ttk.Button(
        button_frame,
        text=text,
        command=command,
        style="TButton"
    )
    btn.pack(side="left", padx=5, expand=True, fill="x")

# Sekcja wyświetlania danych z bardziej wyrazistymi ramkami
original_data_frame = ttk.LabelFrame(root, text="Dane oryginalne", style="TLabelframe")
original_data_frame.pack(fill="both", expand=True, padx=15, pady=10)

processed_data_frame = ttk.LabelFrame(root, text="Dane przetworzone", style="TLabelframe")
processed_data_frame.pack(fill="both", expand=True, padx=15, pady=10)

# Uruchomienie aplikacji
root.mainloop()
