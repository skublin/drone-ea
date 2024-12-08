import re
import csv

def convert_train_output_to_csv(input_path, output_path):
    """
    Konwertuje plik tekstowy z danymi wyjściowymi treningu na plik CSV.
    
    :param input_path: Ścieżka do pliku wejściowego (.txt).
    :param output_path: Ścieżka do pliku wyjściowego (.csv).
    """
    # Regularne wyrażenie do wyodrębniania danych
    regex = re.compile(
        r"Generation: (\d+), Solution: (\d+), Collected Targets: (\d+), Fitness score: ([0-9.]+), Time: ([0-9.]+)"
    )

    data = []

    # Odczyt i parsowanie pliku wejściowego
    with open(input_path, 'r') as file:
        for line in file:
            match = regex.match(line)
            if match:
                generation, solution, targets, fitness, time = match.groups()
                data.append({
                    "Generation": int(generation),
                    "Solution": int(solution),
                    "Collected Targets": int(targets),
                    "Fitness Score": float(fitness),
                    "Time": float(time)
                })

    # Zapis danych do pliku CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ["Generation", "Solution", "Collected Targets", "Fitness Score", "Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Ścieżki do plików wejściowego i wyjściowego
input_file_path = "train_output.txt"  # Ścieżka do pliku wejściowego
output_file_path = "train_output.csv"  # Ścieżka do pliku wyjściowego

# Konwersja pliku
convert_train_output_to_csv(input_file_path, output_file_path)
print(f"Plik został zapisany jako: {output_file_path}")
