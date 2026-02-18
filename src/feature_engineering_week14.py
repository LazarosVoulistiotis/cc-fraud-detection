# src/feature_engineering_week14.py
"""
1.Φορτώνεις splits από Week 13.
2.Κάνεις baseline run.
3.Κάνεις engineered run.
4.Συγκρίνεις."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Θα αποθηκεύσουμε quantile thresholds, Θα αποθηκεύσουμε bin edges, Θα έχουμε scaler object
class AmountFeatureEngineer:
    def __init__(self, use_scaled: bool = True):
        self.use_scaled = use_scaled
        self.q90 = None
        self.q95 = None
        self.q99 = None
        self.bins = None
        self.scaler = StandardScaler()
# Δεν υπολογίζουμε quantiles εδώ γιατί δεν έχουμε δει training data ακόμα. 
# Quantiles πρέπει να υπολογιστούν μόνο στο training set. Αν τα υπολογίζαμε εδώ, χωρίς έλεγχο, θα είχαμε leakage.

    def fit(self, X):
        # compute thresholds from TRAIN ONLY
        self.q90 = X["Amount"].quantile(0.90) # Το 90th percentile σημαίνει: 90% των ποσών είναι κάτω από αυτή την τιμή. ΤΟ θελουμε γιατι το fraud συχνά είναι σπάνιο αλλά μεγάλα ποσά
        self.q95 = X["Amount"].quantile(0.95)
        self.q99 = X["Amount"].quantile(0.99)
        self.bins = X["Amount"].quantile([0, 0.5, 0.9, 0.99, 1.0]).values # δημιουργεί 4 ζώνες bins: 0% (min), 50% (median), 90%, 99%, 100% (max). Τα tree models κάνουν splits. Αν το bin είναι ήδη διακριτό, βοηθάει τα splits να γίνουν καθαρότερα.
        self.scaler.fit(X[["Amount"]]) # Ο scaler μαθαίνει το mean και std από το training set. Αυτά θα χρησιμοποιηθούν για να μετασχηματίσουμε και τα test δεδομένα με τον ίδιο τρόπο, χωρίς leakage.

    def transform(self, X):
        X = X.copy() # (Fit ΜΟΝΟ στο training set.)

        # log
        X["log_amount"] = np.log1p(X["Amount"]) # log1p = log(1 + x) για να χειριστούμε τα μηδενικά ποσά χωρίς να έχουμε -inf. Το log βοηθάει να μειώσουμε την επίδραση των μεγάλων ποσών και να κάνουμε την κατανομή πιο κανονική, κάτι που μπορεί να βοηθήσει τα μοντέλα να μάθουν καλύτερα.

        # threshold flags
        X["amount_gt_90"] = (X["Amount"] > self.q90).astype(int) # Αυτό δημιουργεί binary feature.Πολύ σημαντικό insight: Binary features σε tree models είναι ισχυρά. Είναι καθαρά decision rules. “Is transaction in top 10% amount?” Αυτό είναι σχεδόν business rule.
        X["amount_gt_95"] = (X["Amount"] > self.q95).astype(int)
        X["amount_gt_99"] = (X["Amount"] > self.q99).astype(int)

        # binning (κατηγοριοποίηση)
        X["amount_bin"] = np.digitize(X["Amount"], self.bins[1:-1]) # Το digitize βάζει κάθε ποσό σε κατηγορία.

        # scaling
        if self.use_scaled:
            X["amount_scaled"] = self.scaler.transform(X[["Amount"]])

        return X

# Παρόμοια λογική με το AmountFeatureEngineer, αλλά για την ώρα της συναλλαγής. Η ώρα μπορεί να έχει σημαντική επίδραση στο αν μια συναλλαγή είναι fraud ή όχι. Πολλά frauds συμβαίνουν τη νύχτα, όταν οι άνθρωποι δεν είναι προσεκτικοί.
class TimeFeatureEngineer:

    def transform(self, X):
        X = X.copy()
        # Η ώρα της συναλλαγής μπορεί να είναι σημαντική. Πολλά frauds συμβαίνουν τη νύχτα, όταν οι άνθρωποι δεν είναι προσεκτικοί. Το Time feature είναι σε δευτερόλεπτα από την αρχή της ημέρας. Για να πάρουμε την ώρα, διαιρούμε με 3600 (δευτερόλεπτα ανά ώρα) και παίρνουμε το υπόλοιπο με 24 για να έχουμε την ώρα της ημέρας.
        X["hour"] = ((X["Time"] / 3600) % 24).astype(int)
        # Cyclical Encoding. (Cyclical encoding is a data preprocessing technique that transforms periodic features (e.g., hours, days, months, angles) into sine and cosine components to preserve their continuous, cyclical nature for machine learning models)
        # Το hour = 23 και hour = 0 είναι κοντά χρονικά. Αλλά numerically:23 και 0 είναι μακριά. Sin/cos encoding:Μετατρέπει την ώρα σε σημείο πάνω σε κύκλο. Αυτό αποφεύγει artificial discontinuity.
        X["sin_hour"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["cos_hour"] = np.cos(2 * np.pi * X["hour"] / 24)
        # Γιατί;Fraud συχνά:λιγότερη ανθρώπινη παρακολούθηση, περίεργες ώρες. Αυτό είναι domain-inspired feature.
        X["night_flag"] = ((X["hour"] >= 0) & (X["hour"] < 6)).astype(int)

        return X

class Week14FeatureEngineer:
    """
    Convenience wrapper: fit on train, transform any split.
    """

    def __init__(self, use_amount=True, use_time=True, use_amount_scaled=True):
        self.use_amount = use_amount
        self.use_time = use_time
        self.use_amount_scaled = use_amount_scaled

        self.amount_fe = AmountFeatureEngineer(use_scaled=use_amount_scaled) if use_amount else None
        self.time_fe = TimeFeatureEngineer() if use_time else None

    def fit(self, X_train):
        if self.amount_fe is not None:
            self.amount_fe.fit(X_train)
        return self
    
    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def transform(self, X):
        out = X
        if self.amount_fe is not None:
            out = self.amount_fe.transform(out)
        if self.time_fe is not None:
            out = self.time_fe.transform(out)
        return out