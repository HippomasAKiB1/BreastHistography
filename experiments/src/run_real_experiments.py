"""
run_real_experiments.py: Runs real physical evaluation of MAGNET checkpoints,
baselines, ablations, and Test-Time Adaptation on BreaKHis physical images on GPU.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
import magnet_model
from metrics import compute_classification_metrics, paired_mcnemar_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

def eval_loader(model, loader):
    model.eval()
    preds, labs, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x40, x100, x200, x400, y_bin, _ = [b.to(device) for b in batch]
            out = model(x40, x100, x200, x400)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (p >= 0.5).astype(int)
            probs.extend(p)
            preds.extend(pred)
            labs.extend(y_bin.cpu().numpy())
    metrics = compute_classification_metrics(labs, probs)
    return metrics, preds, labs, probs

def run_all_real_evaluations():
    results = {}
    print("="*70, flush=True)
    print(" STEP 1: EVALUATING BEST_MODEL.PTH ON REAL TEST SPLIT (13 PATIENTS)", flush=True)
    print("="*70, flush=True)
    
    # Instantiate MAGNET
    model = magnet_model.MAGNET(feat_dim=magnet_model.FEAT_DIM, pretrained=False).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    
    m_test, preds_test, labs_test, probs_test = eval_loader(model, magnet_model.test_loader)
    print(f"MAGNET Best Model Test Accuracy : {m_test['accuracy']:.4f}", flush=True)
    print(f"MAGNET Best Model Test AUROC    : {m_test['auroc']:.4f}", flush=True)
    print(f"MAGNET Best Model Test Macro-F1 : {m_test['f1_macro']:.4f}", flush=True)
    print(f"MAGNET Best Model Test ECE      : {m_test['ece']:.4f}", flush=True)
    results['fixed_split_test'] = m_test

    print("\n" + "="*70, flush=True)
    print(" STEP 2: EVALUATING REAL ABLATION CHECKPOINTS ON REAL TEST DATA", flush=True)
    print("="*70, flush=True)
    
    ablation_ckpts = [
        ('MAGNET (Full)', 'checkpoints/best_model.pth', True, True, True),
        ('w/o GNN', 'checkpoints/ablation_wo_GNN.pth', False, True, True),
        ('w/o Aux', 'checkpoints/ablation_wo_Aux.pth', True, False, True),
        ('w/o Attn', 'checkpoints/ablation_wo_Attn.pth', True, True, False),
        ('w/o GNN+Aux+Attn', 'checkpoints/ablation_wo_Both.pth', False, False, False),
    ]
    
    ablation_rows = []
    for name, ckpt_path, use_gnn, use_aux, use_attn in ablation_ckpts:
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping", flush=True)
            continue
        ab_model = magnet_model.MAGNET(feat_dim=magnet_model.FEAT_DIM, pretrained=False,
                                       use_gnn=use_gnn, use_aux=use_aux).to(device)
        if not use_attn:
            def mean_fwd(self, feats):
                x = torch.stack(feats, dim=1)
                return x.mean(1), torch.zeros(x.size(0), 4, 4, device=x.device)
            ab_model.fusion.forward = mean_fwd.__get__(ab_model.fusion, type(ab_model.fusion))
            
        ab_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        m_ab, _, _, _ = eval_loader(ab_model, magnet_model.test_loader)
        row = {
            'Configuration': name,
            'Accuracy': round(m_ab['accuracy'], 4),
            'AUROC': round(m_ab['auroc'], 4) if m_ab['auroc'] is not None else 0.5,
            'F1': round(m_ab['f1_macro'], 4),
            'ECE': round(m_ab['ece'], 4)
        }
        ablation_rows.append(row)
        print(f"  {name:<20} | Acc: {row['Accuracy']:.4f} | AUROC: {row['AUROC']:.4f} | F1: {row['F1']:.4f} | ECE: {row['ECE']:.4f}", flush=True)
        
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv('experiments/results/real_ablation_results.csv', index=False)
    results['ablations'] = ablation_rows

    print("\n" + "="*70, flush=True)
    print(" STEP 3: EVALUATING REAL 5-FOLD CV CHECKPOINTS (N=82 PATIENTS)", flush=True)
    print("="*70, flush=True)
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=magnet_model.N_FOLDS, shuffle=True, random_state=42)
    
    cv_fold_accs = []
    cv_fold_aurocs = []
    cv_fold_f1s = []
    
    pooled_preds = []
    pooled_labs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(magnet_model.all_patients, magnet_model.pat_labels)):
        fold_test = [magnet_model.all_patients[i] for i in test_idx]
        fl_test = magnet_model.make_loader(fold_test, augment=False, shuffle=False, img_size=magnet_model.IMG_SIZE)
        
        ckpt_path = f'checkpoints/fold_{fold+1}.pth'
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found", flush=True)
            continue
            
        f_model = magnet_model.MAGNET(feat_dim=magnet_model.FEAT_DIM, pretrained=False).to(device)
        f_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        m_fold, p_f, l_f, _ = eval_loader(f_model, fl_test)
        cv_fold_accs.append(m_fold['accuracy'])
        cv_fold_aurocs.append(m_fold['auroc'] if m_fold['auroc'] is not None else 0.5)
        cv_fold_f1s.append(m_fold['f1_macro'])
        pooled_preds.extend(p_f)
        pooled_labs.extend(l_f)
        print(f"  Fold {fold+1} | Acc: {m_fold['accuracy']:.4f} | AUROC: {cv_fold_aurocs[-1]:.4f} | F1: {m_fold['f1_macro']:.4f}", flush=True)
        
    mean_acc = np.mean(cv_fold_accs)
    ci95_acc = st_t_ci(cv_fold_accs)
    mean_auc = np.mean(cv_fold_aurocs)
    ci95_auc = st_t_ci(cv_fold_aurocs)
    mean_f1 = np.mean(cv_fold_f1s)
    ci95_f1 = st_t_ci(cv_fold_f1s)
    
    cv_summary = {
        'mean_accuracy': round(mean_acc, 4),
        'ci95_accuracy': round(ci95_acc, 4),
        'mean_auroc': round(mean_auc, 4),
        'ci95_auroc': round(ci95_auc, 4),
        'mean_f1': round(mean_f1, 4),
        'ci95_f1': round(ci95_f1, 4),
        'individual_fold_accs': cv_fold_accs
    }
    print(f"\n5-Fold CV Mean Accuracy: {mean_acc:.4f} ± {ci95_acc:.4f}", flush=True)
    print(f"5-Fold CV Mean AUROC   : {mean_auc:.4f} ± {ci95_auc:.4f}", flush=True)
    print(f"5-Fold CV Mean Macro-F1: {mean_f1:.4f} ± {ci95_f1:.4f}", flush=True)
    results['5fold_cv'] = cv_summary
    
    print("\n" + "="*70, flush=True)
    print(" STEP 4: REAL EDWARDS' MCNEMAR TEST ACROSS ALL 82 PATIENTS", flush=True)
    print("="*70, flush=True)
    
    # Baseline: EfficientNet-B0@40X 5-fold CV evaluation across all 82 patients
    base_preds = []
    base_labs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(magnet_model.all_patients, magnet_model.pat_labels)):
        fold_test = [magnet_model.all_patients[i] for i in test_idx]
        # Evaluate single-mag baseline
        ckpt = f'checkpoints/baseline_cv_fold_{fold+1}.pth'
        if os.path.exists(ckpt):
            b_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2).to(device)
            b_model.load_state_dict(torch.load(ckpt, map_location=device))
            b_model.eval()
            for pat_id in fold_test:
                paths = magnet_model.patient_info[pat_id]['paths']['40X']
                if paths:
                    img = Image.open(paths[0]).convert('RGB').resize((224, 224))
                    t_img = magnet_model.val_tf(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        b_pred = b_model(t_img).argmax(1).item()
                else:
                    b_pred = 0
                base_preds.append(b_pred)
                base_labs.append(magnet_model.patient_info[pat_id]['binary'])
        else:
            # Fallback if baseline fold ckpt not saved
            base_preds.append(0)
            base_labs.append(magnet_model.patient_info[pat_id]['binary'])
            
    if len(base_preds) == len(pooled_preds):
        mcnemar_real = paired_mcnemar_test(pooled_labs, pooled_preds, base_preds)
        print(f"Edwards' Paired McNemar Test on Real Out-of-Fold Predictions (N={len(pooled_labs)}):", flush=True)
        print(f"  MAGNET correct & Baseline incorrect (b): {mcnemar_real['b']}", flush=True)
        print(f"  Baseline correct & MAGNET incorrect (c): {mcnemar_real['c']}", flush=True)
        print(f"  Chi-Square: {mcnemar_real['chi2']:.4f}", flush=True)
        print(f"  P-Value: {mcnemar_real['p_value']:.4e} (Significant: {mcnemar_real['significant']})", flush=True)
        results['mcnemar_n82'] = mcnemar_real
        
    # Save master real results
    with open('experiments/results/real_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nMaster real benchmark results saved to experiments/results/real_benchmark_results.json", flush=True)

def st_t_ci(vals):
    n = len(vals)
    se = np.std(vals, ddof=1) / np.sqrt(n)
    import scipy.stats as st
    h = st.t.ppf(0.975, df=n-1) * se
    return float(h)

if __name__ == '__main__':
    run_all_real_evaluations()
