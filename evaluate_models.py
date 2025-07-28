import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def evaluate_models():
    # Load data
    df = pd.read_csv('diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                             random_state=42, scale_pos_weight=scale_pos_weight)
    
    # Train models
    rf_model.fit(X_train_resampled, y_train_resampled)
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Hybrid predictions (average probability)
    hybrid_proba = (rf_pred_proba + xgb_pred_proba) / 2
    hybrid_pred = (hybrid_proba > 0.5).astype(int)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred, y_proba, model_name):
        return {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'ROC AUC': roc_auc_score(y_true, y_proba),
            'Confusion Matrix': confusion_matrix(y_true, y_pred)
        }
    
    metrics = [
        calculate_metrics(y_test, rf_pred, rf_pred_proba, 'Random Forest'),
        calculate_metrics(y_test, xgb_pred, xgb_pred_proba, 'XGBoost'),
        calculate_metrics(y_test, hybrid_pred, hybrid_proba, 'Hybrid')
    ]
    
    # Generate performance report
    performance_report = {
        'metrics': pd.DataFrame(metrics).drop('Confusion Matrix', axis=1),
        'classification_reports': {
            'Random Forest': classification_report(y_test, rf_pred, output_dict=True),
            'XGBoost': classification_report(y_test, xgb_pred, output_dict=True),
            'Hybrid': classification_report(y_test, hybrid_pred, output_dict=True)
        },
        'confusion_matrices': {
            'Random Forest': metrics[0]['Confusion Matrix'],
            'XGBoost': metrics[1]['Confusion Matrix'],
            'Hybrid': metrics[2]['Confusion Matrix']
        },
        'feature_importances': {
            'Random Forest': rf_model.feature_importances_,
            'XGBoost': xgb_model.feature_importances_
        },
        'feature_names': X.columns.tolist()  # Store feature names for display
    }
    
    # Generate and save ROC curve
    plt.figure(figsize=(10, 8))
    
    for model, pred_proba, color, label in [
        ('Random Forest', rf_pred_proba, 'blue', 'Random Forest'),
        ('XGBoost', xgb_pred_proba, 'red', 'XGBoost'),
        ('Hybrid', hybrid_proba, 'green', 'Hybrid')
    ]:
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()
    performance_report['roc_curve'] = 'roc_curves.png'
    
    # Save models and performance report
    with open('performance_report.pkl', 'wb') as f:
        pickle.dump(performance_report, f)
    
    pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
    pickle.dump(xgb_model, open('xgb_model.pkl', 'wb'))
    
    return performance_report

def display_performance_report(report):
    """Display the performance report in a user-friendly format."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS".center(50))
    print("="*50)
    
    # Display metrics table
    print("\n=== Evaluation Metrics ===")
    print(report['metrics'].to_string(index=False))
    
    # Display classification reports
    print("\n=== Classification Reports ===")
    for model_name, cr in report['classification_reports'].items():
        print(f"\n{model_name}:")
        print(pd.DataFrame(cr).transpose().to_string())
    
    # Display confusion matrices
    print("\n=== Confusion Matrices ===")
    for model_name, cm in report['confusion_matrices'].items():
        print(f"\n{model_name}:")
        print(pd.DataFrame(cm, 
                         index=['Actual Negative', 'Actual Positive'],
                         columns=['Predicted Negative', 'Predicted Positive']))
    
    # Display feature importances
    print("\n=== Feature Importances ===")
    fi_df = pd.DataFrame(report['feature_importances'], 
                        index=report['feature_names']).transpose()
    print(fi_df.to_string())
    
    print("\nROC Curve saved to: roc_curves.png")

if __name__ == "__main__":
    print("Evaluating model performance...")
    performance_report = evaluate_models()
    display_performance_report(performance_report)
    print("\nEvaluation complete! Models and performance report saved.")