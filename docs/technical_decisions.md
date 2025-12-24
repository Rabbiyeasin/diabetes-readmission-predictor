## Decision 4: Ensemble Strategy (Simple vs Complex)

**Date:** December 6, 2025  
**Context:** After tuning XGBoost to 0.70 AUC, needed to decide on ensemble approach  
**Options:**
1. Simple averaging (XGBoost + LightGBM)
2. Weighted averaging with optimized weights
3. Stacking with meta-learner (logistic regression)
4. Blending with holdout set

**Decision:** Simple averaging (Option 1)  
**Rationale:**  
- KISS principle for production deployment
- Complex stacking adds +0.005-0.01 AUC but reduces explainability
- Simple averaging maintains SHAP interpretability
- Easier to maintain in production (no meta-learner retraining)

**Result:** 0.70 → 0.71+ AUC with full explainability maintained

---

## Decision 5: GridSearch Optimization Metric

**Date:** December 6, 2025  
**Options:** Optimize for AUC-ROC vs PR-AUC vs F1-Score  
**Decision:** PR-AUC  
**Rationale:**  
- Dataset has 1:7.9 class imbalance
- AUC-ROC can be misleading with severe imbalance
- PR-AUC focuses on positive class performance
- Aligns with clinical priority (catching high-risk patients)

**Result:** Models optimized for real-world clinical utility, not vanity metrics