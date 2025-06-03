import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import joblib
warnings.filterwarnings('ignore')

class MultasPatternRecognition:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.label_encoders = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_data(self, csv_path):
        """Cargar y preparar los datos"""
        self.df = pd.read_csv(csv_path, delimiter=';', encoding='latin-1')
        print(f"Dataset cargado: {self.df.shape[0]} registros, {self.df.shape[1]} columnas")
        return self.df
    
    def preprocess_data(self):
        """Preprocesamiento de datos y ingeniería de características"""
        df = self.df.copy()
        df = df[df['monto'] > 0]
        
        # Convertir fechas
        date_columns = ['fechamulta', 'fechasistema', 'fechaproyeccion', 'fecha_corte']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
        
        # Crear nuevas características temporales
        df['dias_diferencia_multa_sistema'] = (df['fechasistema'] - df['fechamulta']).dt.days
        df['dias_diferencia_multa_proyeccion'] = (df['fechaproyeccion'] - df['fechamulta']).dt.days
        df['ano_multa'] = df['fechamulta'].dt.year
        df['mes_multa'] = df['fechamulta'].dt.month
        df['dia_semana_multa'] = df['fechamulta'].dt.dayofweek
        
        # Crear características de tiempo transcurrido
        fecha_referencia = pd.to_datetime('2024-06-19')
        df['dias_desde_multa'] = (fecha_referencia - df['fechamulta']).dt.days
        
        # Características financieras
        df['monto_total_original'] = df['monto'] + df['interes'].fillna(0) + df['gastos'].fillna(0) + df['costas'].fillna(0)
        df['porcentaje_descuento'] = (df['descuento'].fillna(0) / df['monto']) * 100
        df['ratio_descuento_monto'] = df['descuento'].fillna(0) / df['monto']
        df['diferencia_monto_total'] = df['monto_total_original'] - df['total']
        
        # Características categóricas mejoradas
        df['giro_categoria'] = df['giro'].str.upper().str.strip()
        df['descripcion_categoria'] = df['descripcion'].str.len()  # Longitud de descripción
        df['zona_num'] = pd.to_numeric(df['zona'], errors='coerce')
        
        # Rangos de montos
        df['rango_monto'] = pd.cut(
            df['monto'], 
            bins=[0, 200, 500, 1000, 2000, float('inf')], 
            labels=['Bajo', 'Medio-Bajo', 'Medio', 'Alto', 'Muy Alto']
        ).astype(str)  # 👈 Esta parte evita el error categórico

        
        # Variable objetivo (P = Pagado, A = Activo/No Pagado)
        df['estado_binario'] = (df['estado'] == 'P').astype(int)
        
        # Seleccionar características para el modelo
        feature_columns = [
            'aniomulta', 'zona_num', 'codigodegiro', 'codigomulta', 'monto', 
            'interes', 'gastos', 'costas', 'descuento', 'total',
            'dias_diferencia_multa_sistema', 'dias_diferencia_multa_proyeccion',
            'ano_multa', 'mes_multa', 'dia_semana_multa', 'dias_desde_multa',
            'monto_total_original', 'porcentaje_descuento', 'ratio_descuento_monto',
            'diferencia_monto_total', 'descripcion_categoria', 'giro_categoria',
            'rango_monto'
        ]
        
        # Limpiar datos faltantes
        df = df.dropna(subset=['estado_binario'])
        
        self.processed_df = df
        return df
    
    def prepare_features(self):
        """Preparar características para el modelo"""
        df = self.processed_df
        
        # Variables numéricas
        numeric_features = [
            'aniomulta', 'zona_num', 'codigodegiro',  'monto', 
            'interes', 'gastos', 'costas', 'descuento', 'total',
            'dias_diferencia_multa_sistema', 'dias_diferencia_multa_proyeccion',
            'ano_multa', 'mes_multa', 'dia_semana_multa', 'dias_desde_multa',
            'monto_total_original', 'porcentaje_descuento', 'ratio_descuento_monto',
            'diferencia_monto_total', 'descripcion_categoria'
        ]
        
        # Variables categóricas
        categorical_features = ['giro_categoria', 'rango_monto','codigomulta']
        
        # Crear preprocesador
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Preparar X e y
        feature_cols = numeric_features + categorical_features
        X = df[feature_cols].copy()
        y = df['estado_binario']
        
        # Limpiar valores faltantes
        X = X.fillna(0)
        
        return X, y
    
    def train_models(self, X, y):
        """Entrenar múltiples modelos con 20% para entrenamiento y 80% para evaluación"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, train_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Datos de entrenamiento: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Datos de evaluación: {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")
        
        # Definir modelos
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42
            )
        }
        
        # Entrenar y evaluar modelos
        results = {}
        self.model = self.best_model

        for name, model in models.items():
            print(f"\nEntrenando {name}...")
            
            # Crear pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Entrenar
            pipeline.fit(X_train, y_train)
            
            # Predecir
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Evaluar
            accuracy = accuracy_score(y_test, y_pred)
            
            # Validación cruzada (ajustada para dataset pequeño de entrenamiento)
            cv_folds = min(3, len(X_train) // 10)  # Ajustar folds según tamaño de datos
            if cv_folds >= 2:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy')
            else:
                cv_scores = np.array([accuracy])  # Si es muy pequeño, usar accuracy simple
            
            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'y_test': y_test
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        self.X_test = X_test
        self.y_test = y_test
        
        # Seleccionar mejor modelo
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nMejor modelo: {best_model_name}")
        
        return results
    
    def get_feature_importance(self):
        """Obtener importancia de características"""
        if self.best_model_name in ['RandomForest', 'GradientBoosting']:
            # Obtener nombres de características después del preprocesamiento
            feature_names = []
            
            # Características numéricas
            numeric_features = [
                'aniomulta', 'zona_num', 'codigodegiro', 'codigomulta', 'monto', 
                'interes', 'gastos', 'costas', 'descuento', 'total',
                'dias_diferencia_multa_sistema', 'dias_diferencia_multa_proyeccion',
                'ano_multa', 'mes_multa', 'dia_semana_multa', 'dias_desde_multa',
                'monto_total_original', 'porcentaje_descuento', 'ratio_descuento_monto',
                'diferencia_monto_total', 'descripcion_categoria'
            ]
            
            feature_names.extend(numeric_features)
            
            # Características categóricas (después de OneHotEncoder)
            cat_encoder = self.best_model.named_steps['preprocessor'].named_transformers_['cat']
            if hasattr(cat_encoder, 'get_feature_names_out'):
                cat_features = cat_encoder.get_feature_names_out(['giro_categoria', 'rango_monto'])
                feature_names.extend(cat_features)
            
            # Importancia
            importances = self.best_model.named_steps['classifier'].feature_importances_
            
            # Crear DataFrame de importancia
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            return importance_df
        
        return None
    
    def evaluation_report(self):
        """Reporte detallado de evaluación con métricas de precisión"""
        print("\n" + "="*80)
        print("REPORTE DE EVALUACIÓN - ENTRENAMIENTO 20% vs EVALUACIÓN 80%")
        print("="*80)
        
        # Información del dataset
        total_samples = len(self.X_test) + len(self.processed_df) - len(self.X_test) / 0.8 * 0.2
        train_samples = int(total_samples * 0.2)
        test_samples = len(self.X_test)
        
        print(f"\nDistribución de datos:")
        print(f"├── Total de registros: {int(total_samples)}")
        print(f"├── Entrenamiento (20%): {train_samples} registros")
        print(f"└── Evaluación (80%): {test_samples} registros")
        
        print(f"\nDistribución de clases en evaluación:")
        unique, counts = np.unique(self.y_test, return_counts=True)
        for i, (clase, count) in enumerate(zip(unique, counts)):
            estado = "Pagado" if clase == 1 else "No Pagado"
            print(f"├── {estado}: {count} registros ({count/len(self.y_test)*100:.1f}%)")
        
        # Evaluar cada modelo
        print(f"\n{'='*80}")
        print("MÉTRICAS DE PRECISIÓN POR MODELO")
        print("="*80)
        
        model_performance = []
        
        for name, results in self.models.items():
            print(f"\n🔹 {name.upper()}")
            print("-" * 50)
            
            y_pred = results['predictions']
            y_prob = results['probabilities']
            
            # Métricas básicas
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Calcular métricas manualmente para mayor control
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            precision_pagado = precision_score(self.y_test, y_pred, pos_label=1, zero_division=0)
            precision_no_pagado = precision_score(self.y_test, y_pred, pos_label=0, zero_division=0)
            
            recall_pagado = recall_score(self.y_test, y_pred, pos_label=1, zero_division=0)
            recall_no_pagado = recall_score(self.y_test, y_pred, pos_label=0, zero_division=0)
            
            f1_pagado = f1_score(self.y_test, y_pred, pos_label=1, zero_division=0)
            f1_no_pagado = f1_score(self.y_test, y_pred, pos_label=0, zero_division=0)
            
            try:
                auc_score = roc_auc_score(self.y_test, y_prob)
            except:
                auc_score = 0
            
            print(f"📊 PRECISIÓN GENERAL: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"📊 AUC-ROC Score: {auc_score:.4f}")
            print(f"📊 CV Score (Entrenamiento): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            
            print(f"\n🎯 MÉTRICAS POR CLASE:")
            print(f"   Multas PAGADAS:")
            print(f"   ├── Precisión: {precision_pagado:.4f} ({precision_pagado*100:.2f}%)")
            print(f"   ├── Recall: {recall_pagado:.4f} ({recall_pagado*100:.2f}%)")
            print(f"   └── F1-Score: {f1_pagado:.4f}")
            
            print(f"   Multas NO PAGADAS:")
            print(f"   ├── Precisión: {precision_no_pagado:.4f} ({precision_no_pagado*100:.2f}%)")
            print(f"   ├── Recall: {recall_no_pagado:.4f} ({recall_no_pagado*100:.2f}%)")
            print(f"   └── F1-Score: {f1_no_pagado:.4f}")
            
            # Matriz de confusión detallada
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            
            print(f"\n📋 MATRIZ DE CONFUSIÓN:")
            print(f"   ┌─────────────────┬──────────┬──────────┐")
            print(f"   │                 │ Predicho │ Predicho │")
            print(f"   │                 │No Pagado │  Pagado  │")
            print(f"   ├─────────────────┼──────────┼──────────┤")
            print(f"   │ Real No Pagado  │   {tn:4d}   │   {fp:4d}   │")
            print(f"   │ Real Pagado     │   {fn:4d}   │   {tp:4d}   │")
            print(f"   └─────────────────┴──────────┴──────────┘")
            
            # Interpretación de resultados
            print(f"\n💡 INTERPRETACIÓN:")
            if accuracy > 0.8:
                print(f"   ✅ Excelente precisión general ({accuracy*100:.1f}%)")
            elif accuracy > 0.7:
                print(f"   🟡 Buena precisión general ({accuracy*100:.1f}%)")
            else:
                print(f"   🔴 Precisión mejorable ({accuracy*100:.1f}%)")
            
            print(f"   • Aciertos totales: {tp + tn} de {len(self.y_test)} casos")
            print(f"   • Errores totales: {fp + fn} de {len(self.y_test)} casos")
            
            # Guardar performance para comparación
            model_performance.append({
                'Modelo': name,
                'Precisión_General': accuracy,
                'Precisión_Pagado': precision_pagado,
                'Precisión_No_Pagado': precision_no_pagado,
                'AUC_Score': auc_score,
                'CV_Mean': results['cv_mean']
            })
        
        # Resumen comparativo
        print(f"\n{'='*80}")
        print("RESUMEN COMPARATIVO DE MODELOS")
        print("="*80)
        
        performance_df = pd.DataFrame(model_performance)
        performance_df = performance_df.sort_values('Precisión_General', ascending=False)
        
        print(f"\n🏆 RANKING POR PRECISIÓN GENERAL:")
        for i, row in performance_df.iterrows():
            rank = performance_df.index.get_loc(i) + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            print(f"{medal} {row['Modelo']}: {row['Precisión_General']:.4f} ({row['Precisión_General']*100:.2f}%)")
        
        # Mejor modelo
        best_model = performance_df.iloc[0]
        print(f"\n🎯 MODELO RECOMENDADO: {best_model['Modelo']}")
        print(f"   └── Logra {best_model['Precisión_General']*100:.2f}% de precisión con solo 20% de datos de entrenamiento")
        
    def detailed_analysis(self):
        """Análisis detallado de patrones"""
        print("\n" + "="*60)
        print("ANÁLISIS DETALLADO DE PATRONES")
        print("="*60)
        """Análisis detallado de patrones"""
        print("\n" + "="*60)
        print("ANÁLISIS DETALLADO DE PATRONES")
        print("="*60)
        
        # Estadísticas generales
        print(f"\nEstadísticas Generales:")
        print(f"Total de multas: {len(self.processed_df)}")
        print(f"Multas pagadas: {sum(self.processed_df['estado_binario'])} ({sum(self.processed_df['estado_binario'])/len(self.processed_df)*100:.1f}%)")
        print(f"Multas no pagadas: {len(self.processed_df) - sum(self.processed_df['estado_binario'])} ({(len(self.processed_df) - sum(self.processed_df['estado_binario']))/len(self.processed_df)*100:.1f}%)")
        
        # Análisis por giro de negocio
        print(f"\nPatrones por tipo de giro:")
        giro_analysis = self.processed_df.groupby('giro_categoria').agg({
            'estado_binario': ['count', 'sum', 'mean']
        }).round(3)
        giro_analysis.columns = ['Total', 'Pagadas', 'Tasa_Pago']
        print(giro_analysis.sort_values('Tasa_Pago', ascending=False))
        
        # Análisis por rangos de monto
        print(f"\nPatrones por rango de monto:")
        monto_analysis = self.processed_df.groupby('rango_monto').agg({
            'estado_binario': ['count', 'sum', 'mean'],
            'monto': 'mean'
        }).round(3)
        monto_analysis.columns = ['Total', 'Pagadas', 'Tasa_Pago', 'Monto_Promedio']
        print(monto_analysis)
        
        # Análisis temporal
        print(f"\nPatrones temporales:")
        temporal_analysis = self.processed_df.groupby('ano_multa').agg({
            'estado_binario': ['count', 'sum', 'mean']
        }).round(3)
        temporal_analysis.columns = ['Total', 'Pagadas', 'Tasa_Pago']
        print(temporal_analysis)
        
        # Métricas detalladas del mejor modelo
        print(f"\nMétricas detalladas del mejor modelo ({self.best_model_name}):")
        best_results = self.models[self.best_model_name]
        print("\nMatriz de Confusión:")
        print(confusion_matrix(self.y_test, best_results['predictions']))
        
        print("\nReporte de Clasificación:")
        print(classification_report(self.y_test, best_results['predictions'], 
                                  target_names=['No Pagado', 'Pagado']))
    
    def predict_new_case(self, case_data):
        """Predecir para un nuevo caso"""
        if self.best_model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train_models primero.")
        
        # Convertir a DataFrame
        case_df = pd.DataFrame([case_data])
        # Asegurar que el valor de 'rango_monto' está correcto
        case_data['rango_monto'] = str(case_data['rango_monto'])

        # Aplicar mismo preprocesamiento
        prediction = self.best_model.predict(case_df)[0]
        probability = self.best_model.predict_proba(case_df)[0]
        
        return {
            'prediccion': 'Pagado' if prediction == 1 else 'No Pagado',
            'probabilidad_pagado': probability[1],
            'probabilidad_no_pagado': probability[0]
        }

# Función principal para ejecutar el análisis
def main():
    """Función principal para ejecutar el análisis completo"""
    
    # Crear instancia del clasificador
    classifier = MultasPatternRecognition()
    
    # Simular datos (reemplazar con tu archivo CSV real)
  # Elimina estas líneas completamente:
# data = " "
# with open('bdmuniate_multas.csv', 'w', encoding='latin-1') as f:
#     f.write(data)

# Y cambia esta línea:
    df = classifier.load_data('bdmuniate_multas.csv')  # Tu archivo real
    
    try:
        # 1. Cargar datos
        print("1. Cargando datos...")
        df = classifier.load_data('bdmuniate_multas.csv')
        
        # 2. Preprocesar datos
        print("2. Preprocesando datos...")
        processed_df = classifier.preprocess_data()
        
        # 3. Preparar características
        print("3. Preparando características...")
        X, y = classifier.prepare_features()
        
        # 4. Entrenar modelos
        print("4. Entrenando modelos...")
        results = classifier.train_models(X, y)
        joblib.dump(classifier.model, 'modelo_entrenado.pkl')


        # 6. Reporte de evaluación detallado
        print("6. Generando reporte de evaluación...")
        performance_df = classifier.evaluation_report()
        
        
        # 8. Análisis detallado de patrones
        classifier.detailed_analysis()
        
        
        return classifier
        
    except Exception as e:
        print(f"Error en el análisis: {str(e)}")
        return None

if __name__ == "__main__":
    clasificador = main()