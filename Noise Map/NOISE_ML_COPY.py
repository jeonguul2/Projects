"""
=============================================================================
Urban Noise Modeling: Spatial Heterogeneity and G-XGBoost Analysis
=============================================================================
Description:
    This script models urban noise levels using Machine Learning (XGBoost)
    and Geographical-XGBoost (G-XGBoost) to uncover spatial non-stationarity
    and the mediating role of street-view imagery (SVI) across different cities.

Author: [당신의 이름]
Date: 2024. XX. XX.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist
import shap
from xgboost import XGBRegressor

# G-XGBoost (Install required: pip install geoxgboost contextily)
try:
    from geoxgboost.geoxgboost import gxgb
    import contextily as ctx
    GXGB_AVAILABLE = True
except ImportError:
    GXGB_AVAILABLE = False
    print("Warning: geoxgboost or contextily not found. Spatial mapping will be limited.")

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION & SETUP
# =============================================================================

# VWorld API Key (For Satellite Basemap)
VWORLD_API_KEY = "D59BC91B-AAE3-331B-8907-03F59689F009"

# Features Configuration
BASE_URBAN = [
    '공업지역', '상업지역', '공공시설', 'pop_POP', 'worker_WOR',
    'bus_dist', 'rail_dist', 'NDVI_mean', 'NDBI_mean',
    '지상층수', '건물비율', 'Highway', 'LANES',
]
SVI_FEATURES = ['surf_w', 'veg_w', 'sky_w', 'infra_w']
ALL_FEATURES = BASE_URBAN + SVI_FEATURES
TARGET = 'leq'
CITY_COL = 'bnd'

CITY_NAMES = {1: 'Incheon', 2: 'Daejeon', 3: 'Ulsan'}
CITY_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}

FEAT_NAME_EN = {
    '공업지역': 'Industrial', '상업지역': 'Commercial', '공공시설': 'Public Facility',
    'pop_POP': 'Population', 'worker_WOR': 'Workers',
    'bus_dist': 'Bus Distance', 'rail_dist': 'Rail Distance',
    'NDVI_mean': 'NDVI', 'NDBI_mean': 'NDBI',
    '지상층수': 'Building Floors', '건물비율': 'Building Coverage',
    'Highway': 'Highway', 'LANES': 'LANES',
    'surf_w': 'Surface (SVI)', 'veg_w': 'Vegetation (SVI)',
    'sky_w': 'Sky View (SVI)', 'infra_w': 'Infrastructure (SVI)',
}

def setup_korean_font():
    """Sets up Korean font for matplotlib visualization."""
    font_candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return
    # Fallback to local Windows path
    win_font = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(win_font):
        fm.fontManager.addfont(win_font)
        prop = fm.FontProperties(fname=win_font)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 2. DATA PREPARATION
# =============================================================================

def load_and_prepare_data(file_path):
    """Loads GeoPackage/Shapefile and translates feature names to English."""
    gdf = gpd.read_file(file_path)
    
    # Translate features for modeling and plotting
    X_en = gdf[ALL_FEATURES].copy()
    X_en.columns = [FEAT_NAME_EN.get(c, c) for c in X_en.columns]
    
    # Ensure Coordinates for G-XGBoost
    gdf['X_coord'] = gdf.geometry.centroid.x
    gdf['Y_coord'] = gdf.geometry.centroid.y
    Coords = gdf[['X_coord', 'Y_coord']].copy()
    
    return gdf, X_en, gdf[TARGET], Coords


# =============================================================================
# 3. XGBOOST & SHAP ANALYSIS (CITY-SPECIFIC)
# =============================================================================

def run_city_specific_shap(gdf, X_en, y):
    """Runs city-specific XGBoost models and generates SHAP visualizations."""
    print("\n--- Running City-Specific XGBoost & SHAP Analysis ---")
    en_features = list(X_en.columns)
    
    city_sv = {}
    city_X = {}
    all_importance = np.zeros(len(en_features))
    
    # 1. Train models and compute SHAP
    for city_id in sorted(gdf[CITY_COL].unique()):
        mask = gdf[CITY_COL] == city_id
        X_city = X_en[mask]
        y_city = y[mask]
        
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                             reg_alpha=1, random_state=42, verbosity=0)
        model.fit(X_city, y_city)
        
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_city)
        
        city_sv[city_id] = sv
        city_X[city_id] = X_city
        all_importance += np.abs(sv).mean(axis=0)
    
    # 2. Plot: SHAP Beeswarm
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for idx, city_id in enumerate(sorted(gdf[CITY_COL].unique())):
        plt.sca(axes[idx])
        shap.summary_plot(city_sv[city_id], city_X[city_id], show=False, max_display=12)
        axes[idx].set_title(f'{CITY_NAMES[city_id]} (n={len(city_X[city_id])})', fontsize=14, fontweight='bold')
    plt.suptitle('XGBoost SHAP: City-Specific Beeswarm', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=300)
    plt.close()

    # 3. Plot: SHAP Dependence (Top 3)
    top3_idx = np.argsort(all_importance)[::-1][:3]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for row, feat_idx in enumerate(top3_idx):
        for col, city_id in enumerate(sorted(gdf[CITY_COL].unique())):
            ax = axes[row, col]
            ax.scatter(city_X[city_id].iloc[:, feat_idx], city_sv[city_id][:, feat_idx],
                       alpha=0.5, s=25, color=CITY_COLORS[city_id])
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(en_features[feat_idx], fontsize=10)
            ax.set_ylabel('SHAP value', fontsize=10)
            ax.set_title(f'{CITY_NAMES[city_id]}: {en_features[feat_idx]}', fontsize=11)
            ax.grid(True, alpha=0.2)
    plt.suptitle('XGBoost SHAP Dependence by City (Top 3 Features)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('shap_dependence.png', dpi=300)
    plt.close()
    print("✓ SHAP visualizations saved.")


# =============================================================================
# 4. G-XGBOOST SPATIAL ANALYSIS
# =============================================================================

def compute_spatial_weights(Coords, bw):
    """Computes spatial distance weights for G-XGBoost."""
    coords_arr = Coords.values
    n = len(coords_arr)
    dist_matrix = cdist(coords_arr, coords_arr, metric='euclidean')
    W = np.zeros((n, n))
    
    for i in range(n):
        sorted_dists = np.sort(dist_matrix[i])
        h = sorted_dists[min(bw, n-1)]
        h = 1e-6 if h == 0 else h
        W[i] = np.exp(-0.5 * (dist_matrix[i] / h) ** 2)
        W[i, i] = 0
    return pd.DataFrame(W)

def run_geoxgboost(gdf, X_en, y, Coords, bw=50):
    """Executes Geographically-XGBoost using spatial weights."""
    if not GXGB_AVAILABLE:
        return
        
    print("\n--- Running G-XGBoost Spatial Analysis ---")
    W = compute_spatial_weights(Coords, bw)
    params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4, 'reg_alpha': 1}
    
    result = gxgb(
        X_en, y, Coords,
        params=params, bw=bw, Kernel='Adaptive', spatial_weights=W,
        feat_importance='gain', alpha_wt_type='fixed', alpha_wt=1,
        test_size=0.3, seed=42, n_splits=5, path_save=False
    )
    
    pred_df = result.get('Prediction')
    if pred_df is not None:
        plot_gxgb_variance(pred_df, gdf)
        print("✓ G-XGBoost analysis and variance plots saved.")
    return result

def plot_gxgb_variance(pred_df, gdf):
    """Plots between-city vs within-city spatial variance of feature importance."""
    imp_cols = [c for c in pred_df.columns if c.startswith('Imp_')]
    pred_df = pred_df.copy()
    pred_df[CITY_COL] = gdf[CITY_COL].values
    
    var_data = []
    for col in imp_cols:
        fname = col.replace('Imp_', '')
        city_means, within_stds = [], []
        for city_id in sorted(gdf[CITY_COL].unique()):
            mask = pred_df[CITY_COL] == city_id
            city_vals = pred_df.loc[mask, col]
            city_means.append(city_vals.mean())
            within_stds.append(city_vals.std())
        
        var_data.append({
            'Feature': fname,
            'Between_City': np.std(city_means),
            'Within_City': np.mean(within_stds),
        })
    
    var_df = pd.DataFrame(var_data).sort_values('Between_City', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(var_df))
    width = 0.35
    ax.bar([i-width/2 for i in x], var_df['Between_City'], width, label='Between-City', color='#FF5722')
    ax.bar([i+width/2 for i in x], var_df['Within_City'], width, label='Within-City', color='#2196F3')
    ax.set_xticks(x)
    ax.set_xticklabels(var_df['Feature'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Std Dev of Local Importance', fontsize=11)
    ax.set_title('G-XGBoost: Spatial Variance Decomposition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('gxgb_variance_decomposition.png', dpi=300)
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize fonts
    setup_korean_font()
    
    # Adjust this path as necessary relative to your GitHub repository structure
    DATA_PATH = r"noise_f.shp" 
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset {DATA_PATH} not found. Please verify the path.")
    else:
        # Load and preprocess
        gdf, X_en, y, Coords = load_and_prepare_data(DATA_PATH)
        
        # Run standard XGBoost SHAP Analysis
        run_city_specific_shap(gdf, X_en, y)
        
        # Run G-XGBoost Spatial Framework
        run_geoxgboost(gdf, X_en, y, Coords, bw=50)
        
        print("\nAll modules executed successfully. Visualizations are saved in the current directory.")