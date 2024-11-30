import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
            show_csv(df)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się załadować pliku: {e}")
    else:
        messagebox.showwarning("Brak pliku", "Nie wybrano żadnego pliku.")

# Funkcja do wyświetlania zawartości CSV w tabeli
def show_csv(data):
    for widget in data_frame.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(data_frame, columns=list(data.columns), show='headings')
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")
    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))
    tree.pack(fill="both", expand=True)

# Funkcja do wykonania imputacji kNN
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

# Funkcja do wykonania normalizacji
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

# Funkcja do wykonania standaryzacji
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

# Funkcja do wykonania dyskretyzacji
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

# Tworzenie głównego okna aplikacji
root = tk.Tk()
root.title("Przetwarzanie danych CSV i uczenie modelu")
root.geometry("900x700")

# Sekcja przycisków
button_frame = tk.Frame(root)
button_frame.pack(side="top", fill="x", pady=10)

load_button = tk.Button(button_frame, text="Załaduj plik CSV", command=load_csv, font=('Arial', 12))
load_button.pack(side="left", padx=5)

knn_button = tk.Button(button_frame, text="Imputacja kNN", command=apply_knn_imputer, font=('Arial', 12))
knn_button.pack(side="left", padx=5)

normalize_button = tk.Button(button_frame, text="Normalizacja", command=apply_normalization, font=('Arial', 12))
normalize_button.pack(side="left", padx=5)

standardize_button = tk.Button(button_frame, text="Standaryzacja", command=apply_standardization, font=('Arial', 12))
standardize_button.pack(side="left", padx=5)

discretize_button = tk.Button(button_frame, text="Dyskretyzacja", command=apply_discretization, font=('Arial', 12))
discretize_button.pack(side="left", padx=5)

# Sekcja wyświetlania danych
data_frame = tk.Frame(root)
data_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Sekcja legendy
legend_frame = tk.Frame(root)
legend_frame.pack(side="bottom", fill="x", pady=10)

legend_text = """Legenda (opis kolumn)
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
legend_label = tk.Label(legend_frame, text=legend_text, font=('Arial', 10), justify="left", anchor="w")
legend_label.pack(fill="x", padx=10)

# Uruchomienie aplikacji
root.mainloop()
