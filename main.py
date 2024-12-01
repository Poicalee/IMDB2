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


# noinspection PyUnresolvedReferences
def apply_knn_imputer():
    global df, df_imputed
    if df is None:
        messagebox.showwarning("Błąd", "Załaduj najpierw plik CSV!")
        return
    knn_imputer = KNNImputer(n_neighbors=3)
    df_imputed = pd.DataFrame(
        knn_imputer.fit_transform(df.drop('porzucenie', axis=1)),
        columns=df.columns[:-1]
    )
    df_imputed['porzucenie'] = df['porzucenie']
    messagebox.showinfo("Sukces", "Imputacja kNN zakończona!")
    show_csv(df_imputed)


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
    global df_discretized

    if df_discretized is None:
        messagebox.showwarning("Błąd", "Najpierw wykonaj dyskretyzację!")
        return

    # Przygotowanie danych do klasyfikacji
    X = df_discretized.drop('porzucenie', axis=1)
    y = df_discretized['porzucenie']

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tworzenie modelu Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predykcja
    y_pred = rf.predict(X_test)

    # Obliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)
    messagebox.showinfo("Sukces", f"Dokładność klasyfikacji Random Forest: {accuracy * 100:.2f}%")

    # Uaktualnienie decyzji na podstawie predykcji Random Forest
    df_discretized['porzucenie'] = rf.predict(X)

    # Wyświetlenie zaktualizowanych danych
    show_csv(df_discretized)


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
