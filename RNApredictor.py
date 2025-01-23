import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/Users/hector/Desktop/DataScience/final proyect/train_data.csv'
df = pd.read_csv(file_path)

#Distribución de la composición de bases
base_counts = df['sequence'].apply(lambda x: pd.Series(list(x)).value_counts()).sum()
base_counts.plot(kind='bar')
plt.xlabel('Base', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Count', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.title('Base Composition', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
# Save the plot
plt.savefig('Base Composition.png')
plt.show()

#aquí la longitud de las secuencias de ARN
sequence_lengths = df['sequence'].apply(len)
sns.histplot(sequence_lengths, kde=True)
plt.xlabel('Sequence Length', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Frequency', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.title('Sequence Length Distribution', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
# Save the plot
plt.savefig('Sequence Length Distribution.png')
plt.show()

#gráfico de queso de los tipos de experimentos --> No se si merece la pena hacerlo
experiment_type_counts = df['experiment_type'].value_counts()
plt.pie(experiment_type_counts, labels=experiment_type_counts.index, autopct='%1.1f%%')
plt.title('Experiment Type Distribution', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
# Save the plot
plt.savefig('Experiment Type Distribution.png')
plt.show()

#Signal-to-noise scatter plot --> por lo que he visto es bastante útil en bioinformática para comparar lo buenas que son las muestras en este caso lo buena que es cada secuencia de ARN.
plt.scatter(df['signal_to_noise'], df['reads'])
plt.xlabel('Signal to Noise', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Reads', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.title('Signal to Noise vs Reads', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
# Save the plot
plt.savefig('Signal to Noise vs Reads.png')
plt.show()

#Esto es un heatmap de la reactividad que por lo visto nos ayuda a conocer que parte de las secuencias es la que tiene más importancia en la funcion de la molécula
# Select only the first 50 reactivity columns
reactivity_columns = [col for col in df.columns if col.startswith('reactivity')][:50]
reactivity_data = df[reactivity_columns]

# Convert the reactivity data to a numpy array
reactivity_array = reactivity_data.values

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(reactivity_array, cmap='viridis', cbar=True, xticklabels=20, yticklabels=False)
plt.xlabel('Position', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Sample', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.title('Reactivity Heatmap (First 50 Columns)', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
# Save the plot
plt.savefig('Reactivity Heatmap (First 50 Columns).png')
plt.show()


#Aquí ya empezamos con el modelo como tal
#vamos a usar parámetros de Sensivity, Positive Predictive Value, MAtthews Correlation Coeficient and F-measure.

#Sensivity para la porporcion de TP que nos dice la cantidad de verdaderas parejas de bases.
def sensitivity(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)

true_positives = 100  
false_negatives = 20  

print(f"True Positives: {true_positives}")
print(f"False Negatives: {false_negatives}")

sensitivity_score = sensitivity(true_positives, false_negatives)
print(f"Sensitivity: {sensitivity_score}")

#PPV = Precisión --> mide la proporcion de TP frente a los TP+FP
def ppv(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

true_positives = 100  
false_positives = 30  

ppv_score = ppv(true_positives, false_positives)
print(f"Positive Predictive Value (PPV): {ppv_score}")

#Coeficiente de correlacion --> tiene en cuenta TP, TN, FP, FN, cuanto más cercano 1 mejor.
def mcc(true_positives, true_negatives, false_positives, false_negatives):
    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator = ((true_positives + false_positives) * (true_positives + false_negatives) * 
                   (true_negatives + false_positives) * (true_negatives + false_negatives)) ** 0.5
    return numerator / denominator if denominator != 0 else 0


true_positives = 100  
true_negatives = 50   
false_positives = 30  
false_negatives = 20  

mcc_score = mcc(true_positives, true_negatives, false_positives, false_negatives)
print(f"Matthews Correlation Coefficient (MCC): {mcc_score}")

#F-measure --> Sensivity 
def f_measure(true_positives, false_positives, false_negatives):
    precision = ppv(true_positives, false_positives)
    recall = sensitivity(true_positives, false_negatives)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

true_positives = 100  
false_positives = 30  
false_negatives = 20  

f_measure_score = f_measure(true_positives, false_positives, false_negatives)
print(f"F-measure (F1 Score): {f_measure_score}")

#Comparativa de todos los parámetros calculados
import plotly.graph_objects as go

# Define the performance measures
labels = ['Sensitivity', 'Positive Predictive Value (PPV)', 'Matthews Correlation Coefficient (MCC)', 'F-measure (F1 Score)']
scores = [sensitivity_score, ppv_score, mcc_score, f_measure_score]

# Create a horizontal bar plot
fig = go.Figure(data=[go.Bar(
    y=labels,
    x=scores,
    orientation='h',
    marker=dict(color=['blue', 'green', 'red', 'purple'])
)])

fig.update_layout(
    title='Performance Measures',
    xaxis_title='Score',
    yaxis_title='Metric',
    yaxis=dict(autorange='reversed')
)

fig.show()