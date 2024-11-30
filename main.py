import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

# Inicjalizacja zmiennych globalnych
df = None
df_imputed = None
df_normalized = None
df_standardized = None
df_discretized = None


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


# Funkcja do wyświetlania oryginalnego CSV w tabeli
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

    # Dodanie paska przewijania
    scrollbar = ttk.Scrollbar(original_data_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)


# Funkcja do wyświetlania przetworzonych danych
def show_csv(data):
    for widget in processed_data_frame.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(processed_data_frame, columns=list(data.columns), show='headings', height=10)
    style.configure("Treeview", font=("Consolas", 10), rowheight=25)
    style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")
    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))

    # Dodanie paska przewijania
    scrollbar = ttk.Scrollbar(processed_data_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)


# Funkcje przetwarzania
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


def run_pipeline():
    global df, df_discretized
    if df is None:
        messagebox.showwarning("Błąd", "Załaduj najpierw plik CSV!")
        return

    try:
        knn_imputer = KNNImputer(n_neighbors=3)
        df_imputed = pd.DataFrame(
            knn_imputer.fit_transform(df.drop('porzucenie', axis=1)),
            columns=df.columns[:-1]
        )
        df_imputed['porzucenie'] = df['porzucenie']

        minmax_scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(
            minmax_scaler.fit_transform(df_imputed.drop('porzucenie', axis=1)),
            columns=df_imputed.columns[:-1]
        )
        df_normalized['porzucenie'] = df_imputed['porzucenie']

        standard_scaler = StandardScaler()
        df_standardized = pd.DataFrame(
            standard_scaler.fit_transform(df_normalized.drop('porzucenie', axis=1)),
            columns=df_normalized.columns[:-1]
        )
        df_standardized['porzucenie'] = df_normalized['porzucenie']

        discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
        df_discretized = pd.DataFrame(
            discretizer.fit_transform(df_standardized.drop('porzucenie', axis=1)),
            columns=df_standardized.columns[:-1]
        )
        df_discretized['porzucenie'] = df_standardized['porzucenie']

        messagebox.showinfo("Sukces", "Potok przetwarzania danych zakończony!")
        show_csv(df_discretized)

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

    # Stylizacja okna legendy
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
root.geometry("1200x900")
root.configure(bg="#2c3e50")  # Ciemniejszy, elegancki kolor tła

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
    ("Pokaż legendę", show_legend, "#95a5a6")  # Szary
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