# =====================================================================
# Team 8 - People and HR Analytics | BANA Team Project, PGDM BA 2026
# D4: Machine Learning - Decision Tree and K-Means
# =====================================================================
# WHAT THIS FILE DOES (in plain words)
#   1. Loads the IBM HR employee file.
#   2. Decision Tree : learns simple rules to guess whether an employee
#                      has LOW or HIGH job satisfaction.
#   3. K-Means       : puts similar employees into groups (segments)
#                      so HR can plan one action per group.
#   4. Prints every number you need to fill the yellow [__] slots
#      in Team8_D4_Note.docx.
#
# HOW TO RUN
#   - Keep the CSV in the same folder as this file, then run:
#         python Team8_D4_Machine_Learning_Simple.py
#   - Or paste each "# %%" block into a notebook cell (Jupyter / Databricks).
#   - If the CSV is somewhere else, change DATA_PATH below.
#
# TEAM RULES (from D2 and D3)
#   - We talk about "employees", not "developers" (no developer role in data).
#   - Age, Gender, MaritalStatus are NOT used to train the models.
#     They are used only at the end to check fairness.
#   - JobSatisfaction is NOT given to the tree (it is what we predict).
#   - Results show patterns, NOT causes.
# =====================================================================

# %% STEP 0 - Import the tools we need
import numpy as np                      # numbers and arrays
import pandas as pd                     # tables (DataFrames)
import matplotlib.pyplot as plt         # charts

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer       # treat number / text columns differently
from sklearn.pipeline import Pipeline               # chain steps together
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer            # fill missing values
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, silhouette_score)

SEED = 42   # fixed number so everyone gets the same results every time
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


# %% STEP 1 - Load the data
df = pd.read_csv(DATA_PATH)             # read the file (the CSV itself is never changed)
print("Rows and columns:", df.shape)    # expect (1470, 35)
print("Missing cells   :", df.isna().sum().sum())
print("Duplicate rows  :", df.duplicated().sum())


# %% STEP 2 - Prepare the data
# 2a. Turn the 1-4 satisfaction score into two groups: Low (1,2) and High (3,4)
df["JobSat_Group"] = np.where(df["JobSatisfaction"] >= 3, "High", "Low")

# 2b. Remove columns that are useless: an ID and columns that are the same for everyone
df = df.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])

# 2c. Age bands - used ONLY for the fairness check at the end
df["AgeBand"] = pd.cut(df["Age"], bins=[0, 29, 44, 200],
                       labels=["Under 30", "30-44", "45 and over"])

print("\nSatisfaction groups:")
print(df["JobSat_Group"].value_counts())
print((df["JobSat_Group"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")
# Low is the smaller group, so we must look at RECALL for Low, not only accuracy.


# =====================================================================
# TECHNIQUE 1 - DECISION TREE
# A tree asks a series of yes/no questions (e.g. "Overtime = Yes?")
# and ends with a guess: Low or High.
# =====================================================================

# %% STEP 3 - Choose the columns the tree may use
# Columns we keep OUT of the tree and why:
leave_out = [
    "JobSatisfaction", "JobSat_Group",   # this is what we predict
    "Age", "Gender", "MaritalStatus",    # sensitive - fairness check only
    "AgeBand",                           # made from Age
    "Attrition",                         # background only
]
# Pay-rate columns: drop them if they barely relate to MonthlyIncome (D3 rule)
for col in ["DailyRate", "HourlyRate", "MonthlyRate"]:
    if abs(df[col].corr(df["MonthlyIncome"])) < 0.10:
        leave_out.append(col)
        print("Dropped weak pay-rate column:", col)

X = df.drop(columns=leave_out)           # inputs (the clues)
y = df["JobSat_Group"]                   # target (the answer)

number_cols = X.select_dtypes(include="number").columns.tolist()
text_cols = X.select_dtypes(exclude="number").columns.tolist()
print("\nFields given to the tree:", X.shape[1])

# %% STEP 4 - Get the data ready for the tree
# Numbers: fill any gaps with the median.
# Text   : fill gaps with the most common value, then turn each category into 0/1 columns.
prep = ColumnTransformer([
    ("numbers", SimpleImputer(strategy="median"), number_cols),
    ("text", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
     text_cols),
])

# %% STEP 5 - Split into TRAIN (80%) and TEST (20%)
# The tree learns from TRAIN. TEST is kept hidden and used once to judge it honestly.
# stratify=y keeps the Low/High share the same in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
print("Train employees:", len(X_train), "| Test employees:", len(X_test))

# %% STEP 6 - Build the tree and pick its size
# max_depth      : how many questions in a row (small = easy to read, less over-fitting)
# min_samples_leaf: every final box must hold at least this many employees
# class_weight="balanced": gives the smaller Low group extra attention
pipeline = Pipeline([
    ("prep", prep),
    ("tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
])

# Try a few sizes using only the TRAIN data (5-fold cross-validation) and keep the best.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = GridSearchCV(
    pipeline,
    param_grid={"tree__max_depth": [2, 3, 4], "tree__min_samples_leaf": [20, 40]},
    cv=cv, scoring="balanced_accuracy")
search.fit(X_train, y_train)
model = search.best_estimator_
print("\nChosen settings:", search.best_params_)

# %% STEP 7 - Check how good the tree is (on the hidden TEST employees)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
baseline = (y_test == "High").mean()     # accuracy if we simply said "High" for everyone
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

print("\n--- DECISION TREE RESULTS (copy into Section 3 of the note) ---")
print(f"Test accuracy        : {test_acc:.1%}   (baseline 'always High' = {baseline:.1%})")
print(f"Balanced accuracy    : {balanced_accuracy_score(y_test, pred_test):.1%}")
print(f"Low - recall         : {recall_score(y_test, pred_test, pos_label='Low'):.1%}"
      "   <- share of truly-Low employees the tree found")
print(f"Low - precision      : {precision_score(y_test, pred_test, pos_label='Low'):.1%}"
      "   <- of those flagged Low, share that really are Low")
print(f"Low - F1             : {f1_score(y_test, pred_test, pos_label='Low'):.2f}")

# Over-fitting check: a big gap between train and test means the tree "memorised" the train data.
print(f"Train accuracy       : {train_acc:.1%}")
print(f"Train - test gap     : {(train_acc - test_acc) * 100:.1f} points (about 5 or less is fine)")
print(f"5-fold CV accuracy   : {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")

# Confusion matrix: rows = actual, columns = predicted
cm = confusion_matrix(y_test, pred_test, labels=["Low", "High"])
print("\nConfusion matrix (rows actual Low/High, columns predicted Low/High):")
print(pd.DataFrame(cm, index=["Actual Low", "Actual High"],
                   columns=["Pred Low", "Pred High"]))

# %% STEP 8 - What did the tree use? (patterns, NOT causes)
tree = model.named_steps["tree"]
names = model.named_steps["prep"].get_feature_names_out()
names = [n.split("__", 1)[1] for n in names]            # tidy the names

importance = pd.Series(tree.feature_importances_, index=names)
# The text columns were split into many 0/1 columns - add them back to the original field.
def original_field(name):
    for c in text_cols:
        if name.startswith(c + "_"):
            return c
    return name
importance = importance.groupby(original_field).sum().sort_values(ascending=False)
importance = importance[importance > 0]
print("\nMost-used fields in the tree:")
print(importance.round(3).head(8))
print("\nTip: if tenure fields (years at company etc.) are at the top, they may be")
print("standing in for age - mention this proxy risk (D2).")

# Text version of the tree rules
print("\nTree rules:")
print(export_text(tree, feature_names=names, max_depth=4))

# %% STEP 9 - Charts for the tree
plt.figure(figsize=(11, 4.5))
plot_tree(tree, feature_names=names, class_names=list(tree.classes_),
          max_depth=2, filled=True, rounded=True, impurity=False, proportion=True, fontsize=8)
plt.title("Decision tree (top 3 levels)")
plt.savefig("fig_tree.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 3.5))
importance.head(8).iloc[::-1].plot(kind="barh", color="#2F5D8A")
plt.title("Most-used fields in the tree")
plt.tight_layout()
plt.savefig("fig_importance.png", dpi=200)
plt.show()

# %% STEP 10 - Fairness check (D2 plan)
# We predict for EVERY employee using cross-validation (each employee is predicted by a
# tree that never saw them), so each group is big enough to compare.
# Age and Gender were never inputs, but we still check if results differ between groups.
all_pred = pd.Series(cross_val_predict(model, X, y, cv=cv), index=y.index)

rows = []
for label, column in [("Gender", "Gender"), ("Age band", "AgeBand")]:
    for group in df[column].dropna().unique():
        mask = df[column] == group
        actual_low = (y[mask] == "Low")
        pred_low = (all_pred[mask] == "Low")
        rows.append({
            "Type": label, "Group": group, "Employees": int(mask.sum()),
            "Low employees": int(actual_low.sum()),
            "Accuracy": (y[mask] == all_pred[mask]).mean(),
            "Recall (Low)": (actual_low & pred_low).sum() / max(actual_low.sum(), 1),
            "Precision (Low)": (actual_low & pred_low).sum() / max(pred_low.sum(), 1),
        })
fair = pd.DataFrame(rows).sort_values(["Type", "Group"])
print("\n--- FAIRNESS TABLE (copy into the note) ---")
print(fair.round(3).to_string(index=False))
print("Look at 'Recall (Low)': a gap above 10 points between groups needs a comment.")
print("Groups with under 30 Low employees are 'small' - do not make strong claims.")


# =====================================================================
# TECHNIQUE 2 - K-MEANS CLUSTERING
# K-Means puts similar employees together. We do NOT tell it who is
# satisfied - it only looks at pay, level, tenure and workplace ratings.
# =====================================================================

# %% STEP 11 - Choose the columns for clustering
cluster_cols = ["MonthlyIncome", "JobLevel", "TotalWorkingYears", "YearsAtCompany",
                "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
                "NumCompaniesWorked", "DistanceFromHome", "JobInvolvement",
                "WorkLifeBalance", "EnvironmentSatisfaction"]
cdata = df[cluster_cols].copy()
cdata["MonthlyIncome"] = np.log1p(cdata["MonthlyIncome"])   # income is skewed; log evens it out

# Scale every column to the same range so big numbers (income) don't dominate small ones (ratings)
scaled = StandardScaler().fit_transform(cdata)

# %% STEP 12 - How many groups (k)?
# Silhouette score: higher = groups are more clearly separated (max 1).
# Inertia (elbow): lower = tighter groups, but it always falls as k grows.
print("\n--- CHOOSING k ---")
results = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(scaled)
    results.append({"k": k, "inertia": km.inertia_,
                    "silhouette": silhouette_score(scaled, km.labels_)})
k_table = pd.DataFrame(results)
print(k_table.round(3).to_string(index=False))

# k=2 usually just splits "junior" and "senior", which is too coarse to act on.
# So we pick the best silhouette from k=3 upward.
best_k = int(k_table[k_table["k"] >= 3].sort_values("silhouette").iloc[-1]["k"])
print("Chosen k:", best_k)
print("(A silhouette below about 0.25 means the groups overlap - say so in the note.)")

plt.figure(figsize=(6, 3.5))
plt.plot(k_table["k"], k_table["silhouette"], "o-", color="#2F5D8A")
plt.axvline(best_k, color="#D9822B", linestyle="--")
plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette score")
plt.title("Choosing k")
plt.tight_layout()
plt.savefig("fig_choose_k.png", dpi=200)
plt.show()

# %% STEP 13 - Build the final groups and describe them
df["Cluster"] = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit_predict(scaled)

# Number the segments S1, S2... from lowest to highest median income (easier to read)
order = df.groupby("Cluster")["MonthlyIncome"].median().sort_values().index
df["Segment"] = df["Cluster"].map({old: f"S{i + 1}" for i, old in enumerate(order)})

# Profile = the average of each group, in normal units (not scaled)
profile = df.groupby("Segment").agg(
    Employees=("Segment", "size"),
    Median_income=("MonthlyIncome", "median"),
    Avg_job_level=("JobLevel", "mean"),
    Avg_years_at_company=("YearsAtCompany", "mean"),
    Avg_years_since_promotion=("YearsSinceLastPromotion", "mean"),
    Avg_work_life_balance=("WorkLifeBalance", "mean"),
    Pct_overtime=("OverTime", lambda s: (s == "Yes").mean()),
    Pct_High_satisfaction=("JobSat_Group", lambda s: (s == "High").mean()),
)
print("\n--- SEGMENT PROFILE (copy into the note) ---")
print(profile.round(2).to_string())
print("\nCompany-wide High-satisfaction share:",
      f"{(df['JobSat_Group'] == 'High').mean():.1%}")

# %% STEP 14 - Is satisfaction different between segments?
# Satisfaction was NOT used to make the groups; we compare it only now.
from scipy.stats import chi2_contingency
table = pd.crosstab(df["Segment"], df["JobSat_Group"])
chi2, p, _, _ = chi2_contingency(table)
cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
print(f"\nChi-square p-value: {p:.3f} | Cramer's V: {cramers_v:.2f}")
print("p below 0.05 = difference is statistically significant; V shows how big it is")
print("(under 0.1 = negligible/small).")

# %% STEP 15 - One action per segment (YOU fill this in after reading the profile)
# Look for what stands out in each row of the profile table, e.g.
#   lowest High-satisfaction share  -> stay interviews / pulse survey
#   longest since promotion         -> review career paths and promotion timing
#   most overtime / lowest balance  -> review workload, flexible hours
#   newest to the company           -> onboarding and mentoring
#   highest pay and level           -> retention and succession planning
print("\nNow write ONE action for each segment in Section 4 of the note.")
print("Remember: these are patterns in simulated data, not proof of cause.")
