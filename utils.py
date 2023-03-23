from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

def evaluate(y_test, y_pred,model_name):
    if model_name == 'BiGRU_pretrain':
        y_test_subject=y_test[0]
        y_pred_subject=y_pred[0]
        y_test_polarized=y_test[1]
        y_pred_polarized=y_pred[1]
        # evaluate the subject prediction
        subject_acc = accuracy_score(y_test_subject,y_pred_subject)
        subject_f1 = f1_score(y_test_subject,y_pred_subject, average='weighted')

        # evaluate the polarized prediction
        polarized_acc = accuracy_score(y_test_polarized,y_pred_polarized)
        polarized_f1 = f1_score(y_test_polarized,y_pred_polarized, average='weighted')

        # print the evaluation results
        print(f"Subject accuracy: {subject_acc:.4f}, Subject F1: {subject_f1:.4f}")
        print(f"Polarized accuracy: {polarized_acc:.4f}, Polarized F1: {polarized_f1:.4f}")


    elif model_name == 'BiGRU_attention':
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # print the evaluation results
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'AUC: {auc:.4f}')
        print(f'F1 Score: {f1:.4f}')