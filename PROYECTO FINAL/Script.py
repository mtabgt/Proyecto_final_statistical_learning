import datetime
import time
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def train_model(train_data):
    start_time = time.time() # Se realiza el entrenamiento del pipeline de ingeniería de características y modelo
     
    preds = attritions_pipeline.predict(X_test)


    
    accuracy = accuracy_score(y_true, y_pred) # Se obtienen las métricas de enrtenamiento
    roc_auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Se genera el nombre del archivo con la fecha y hora actual
    output_file = f"train_results_{current_time}.txt"

    
    with open(output_file, "w") as f: # Se registran los resultados en el archivo
        f.write(f"Fecha y hora de ejecución: {current_time}\n")
        f.write(f"Tiempo de entrenamiento (segundos): {time.time() - start_time}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        f.write(f"ROC-AUC: {roc_auc}\n")

    print(f"Entrenamiento completado. Los resultados se han guardado en {output_file}")

def predict(test_data, output_file):
    
    
    predictions.to_csv(output_file, index=False) # Se guardan las predicciones en un archivo CSV
    print(f"Predicciones guardadas en {output_file}")

if __name__ == "__main__": # Se obtienen los argumentos de la línea de comandos
    
    import argparse

    parser = argparse.ArgumentParser(description="Script de entrenamiento y predicción")
    parser.add_argument("operation", choices=["train", "predict"], help="Operación a realizar: train o predict")
    parser.add_argument("--train_data", help="Ruta al archivo de datos de entrenamiento (para la operación train)")
    parser.add_argument("--test_data", help="Ruta al archivo de datos de prueba (para la operación predict)")
    parser.add_argument("--output_file", help="Ruta al archivo de salida (para la operación predict)")
    args = parser.parse_args()

    if args.operation == "train":   
        if not args.train_data:
            print("Se requiere el argumento --train_data para la operación train")
        else:
            train_data = pd.read_csv(args.train_data)
            train_model(train_data)
    elif args.operation == "predict":
        if not args.test_data or not args.output_file:
            print("Se requieren los argumentos --test_data y --output_file para la operación predict")
        else:
            test_data = pd.read_csv(args.test_data)
            predict(test_data, args.output_file)
