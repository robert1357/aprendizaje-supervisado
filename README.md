# 📊 Proyecto: Predicción de Cumplimiento de Pago de Multas Municipales en Ate

**Autores**: Roberth Carlos Gonzales Mauricio, Cesia Gomez Flores  
**Institución**: Universidad Nacional del Altiplano, Puno, Perú  

## 🔍 Resumen
Este proyecto aplica **aprendizaje supervisado** para predecir el cumplimiento de pago de multas en el municipio de Ate (Perú), utilizando un dataset de **53,506 registros**. Se compararon 4 modelos de clasificación, destacando **Gradient Boosting** con un **99.27% de precisión**.

## 📂 Dataset
- **Registros**: 53,506  
- **Columnas**: 23 (incluyen tipo de giro, monto de multa, año, etc.)  
- **Distribución de clases**:  
  - Pagadas: 58.3%  
  - No pagadas: 41.7%  

## 🛠️ Preprocesamiento
1. **Manejo de valores faltantes**: Imputación/eliminación según contexto.  
2. **Normalización**: `StandardScaler` para montos.  
3. **Codificación**: `OneHotEncoder` para variables categóricas (ej. tipo de giro).  
4. **División de datos**:  
   - **Entrenamiento**: 20% (10,701 registros).  
   - **Evaluación**: 80% (42,805 registros).  

## 🤖 Modelos Evaluados
| Modelo               | Precisión | AUC-ROC | Recall (No Pagado) | F1-Score |
|----------------------|-----------|---------|--------------------|----------|
| Gradient Boosting 🏆 | 99.27%    | 0.9987  | 99.45%             | 0.9913   |
| Regresión Logística  | 98.54%    | 0.9965  | 99.33%             | 0.9826   |
| SVM                  | 98.50%    | 0.9973  | 99.25%             | 0.9822   |
| Random Forest        | 96.80%    | 0.9966  | 99.83%             | 0.9630   |

## 📊 Patrones Clave
### Por tipo de giro:
- **Mayor tasa de pago**: Comercializadoras (100%).  
- **Menor tasa de pago**: Telecomercio (0%).  

### Por monto:
- **Multas muy altas**: Tasa de pago del 39.6%.  
- **Multas bajas**: Tasa de pago del 66.5%.  

### Temporal:
- **Años con mayor pago**: 1999 (100%), 2015 (77.8%).  
- **Años con menor pago**: 2018 (30.5%), 2005 (35.6%).  

## 📌 Matriz de Confusión (Gradient Boosting)
|                     | Predicho: No Pagado | Predicho: Pagado |
|---------------------|---------------------|------------------|
| **Real: No Pagado** | 17,739              | 98               |
| **Real: Pagado**    | 214                 | 24,754           |

**Errores totales**: 312 (0.73% del total).  

## 🚀 Recomendaciones
- **Priorizar cobranza**: Multas de monto alto y giros con baja tasa histórica de pago.  
- **Optimizar recursos**: Usar el modelo para focalizar esfuerzos en deudores probables.  

## 📂 Repositorio
[Código y datos disponibles aquí](https://github.com/robert1357/aprendizaje-supervisado).  


