# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:55:20 2025

@author: Ele_p
"""

def process_color_data(input_filename):
    """
    Legge un file di input e separa i dati in base al colore (B, R, G)
    in tre file di output diversi. Il colore è identificato dal carattere
    dopo il carattere '_' nella prima colonna.
    """
    # Dizionario per mappare le lettere ai nomi dei file
    output_files = {
        'B': 'calibration_blu_new_scanner.txt',
        'R': 'calibration_red_new_scanner.txt',
        'G': 'calibration_green_new_scanner.txt'
    }
    
    # Dizionari per memorizzare temporaneamente i dati per ogni colore
    color_data = {color: [] for color in output_files.keys()}
    
    try:
        # Leggi il file di input
        with open(input_filename, 'r') as file:
            for line in file:
                # Rimuovi spazi bianchi iniziali e finali
                line = line.strip()
                if line:  # Verifica che la linea non sia vuota
                    # Dividi la riga in colonne (assumendo che siano separate da spazi o tab)
                    columns = line.split()
                    if columns:  # Verifica che ci siano colonne
                        # Prendi la prima colonna e cerca il carattere dopo '_'
                        first_column = columns[0]
                        if '_' in first_column:
                            # Prendi il carattere dopo '_'
                            color = first_column.split('_')[1]
                            if color in color_data:
                                color_data[color].append(line)
    
        # Scrivi i dati nei rispettivi file di output
        for color, filename in output_files.items():
            with open(filename, 'w') as file:
                for line in color_data[color]:
                    file.write(line + '\n')
                    
        print("Elaborazione completata con successo!")
        print(f"Righe processate per colore:")
        for color in color_data:
            print(f"{color}: {len(color_data[color])} righe")
        
    except FileNotFoundError:
        print(f"Errore: Il file {input_filename} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {str(e)}")

# Esempio di utilizzo
if __name__ == "__main__":
    input_file = "calibration_new_scanner.txt"  # Sostituisci con il nome del tuo file
    process_color_data(input_file)