# ============================================
# 完整代码：三分类癌症预测模型分析 - ROC修复版
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("中文显示已启用")
except:
    print("使用英文显示")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, confusion_matrix, roc_curve, auc,
                             roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("所有必要的库已成功导入！")

# ============================================
# 第一部分：数据加载与预处理
# ============================================

def load_and_preprocess_data(file_path):
    """加载数据并进行初步预处理"""
    print("\n" + "="*50)
    print("第一步：数据加载与预处理")
    print("="*50)
    
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        print(f"成功读取文件: {file_path}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None, None, None
    
    print(f"原始数据形状: {df.shape}")
    
    # 定义特征列（16个变量）
    numerical_features = ['Age', 'cervical lesion SUVmax', 'Liver SUV ratio', 
                          'Blood pool ratio', 'Diameter']
    categorical_features = ['N stage', 'M stage', 'Peritoneal metastasis', 
                            'FIGO stage', 'Growth pattern', 'Intrauterine fluid', 
                            'Cyst', 'Ca199', 'CEA', 'Ca125', 'SCC']
    target_col = 'Cancer type'
    
    # 删除目标变量为空的样本
    df = df.dropna(subset=[target_col])
    print(f"删除缺失目标值后形状: {df.shape}")
    
    # 处理数值列
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理分类变量
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'NaN', 'None'], np.nan)
    
    # 填充分类变量的缺失值
    for col in categorical_features:
        if col in df.columns:
            mode_vals = df[col].mode()
            mode_val = mode_vals[0] if len(mode_vals) > 0 else '0'
            df[col] = df[col].fillna(mode_val)
    
    # 填充数值变量的缺失值
    for col in numerical_features:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    df[target_col] = df[target_col].astype(int)
    
    print(f"最终数据形状: {df.shape}")
    print(f"原始目标变量分布:\n{df[target_col].value_counts().sort_index()}")
    
    return df, numerical_features, categorical_features, target_col

# ============================================
# 第二部分：描述性统计分析
# ============================================

def descriptive_statistics(df, numerical_features, categorical_features, target_col):
    """进行描述性统计分析"""
    print("\n" + "="*50)
    print("第二步：描述性统计分析")
    print("="*50)
    
    # 1. 为每个数值变量单独绘制箱线图
    print("\n--- 绘制数值变量箱线图 ---")
    for feature in numerical_features:
        if feature in df.columns:
            plt.figure(figsize=(8, 6))
            data_to_plot = [df[df[target_col] == cat][feature].dropna() 
                           for cat in sorted(df[target_col].unique())]
            
            bp = plt.boxplot(data_to_plot, labels=[f'Type {cat}' for cat in sorted(df[target_col].unique())],
                           patch_artist=True)
            
            # 设置颜色
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
                patch.set_facecolor(color)
            
            plt.title(f'{feature} by Cancer Type', fontsize=14, fontweight='bold')
            plt.ylabel(feature, fontsize=12)
            plt.xlabel('Cancer Type', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            overall_mean = df[feature].mean()
            plt.axhline(y=overall_mean, color='red', linestyle='--', label=f'Overall Mean: {overall_mean:.2f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'boxplot_{feature.replace(" ", "_")}.png', dpi=300)
            plt.show()
            
            print(f"\n{feature} 统计信息:")
            for cat in sorted(df[target_col].unique()):
                subset = df[df[target_col] == cat][feature]
                print(f"  Type {cat}: n={len(subset)}, Mean={subset.mean():.2f}, Std={subset.std():.2f}, "
                      f"Median={subset.median():.2f}")
    
    # 2. 为每个分类变量绘制柱状图
    print("\n--- 绘制分类变量柱状图 ---")
    for feature in categorical_features:
        if feature in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            counts = df[feature].value_counts()
            axes[0].bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
            axes[0].set_xticks(range(len(counts)))
            axes[0].set_xticklabels(counts.index, rotation=45, ha='right')
            axes[0].set_title(f'{feature} - Overall Distribution', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(feature, fontsize=10)
            axes[0].set_ylabel('Count', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            contingency = pd.crosstab(df[feature], df[target_col])
            contingency.plot(kind='bar', ax=axes[1], color=['blue', 'green', 'red'], alpha=0.7)
            axes[1].set_title(f'{feature} - Distribution by Cancer Type', fontsize=12, fontweight='bold')
            axes[1].set_xlabel(feature, fontsize=10)
            axes[1].set_ylabel('Count', fontsize=10)
            axes[1].legend(title='Cancer Type')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'barplot_{feature.replace(" ", "_")}.png', dpi=300)
            plt.show()
    
    print("\n描述性统计分析完成！")

# ============================================
# 第三部分：数据预处理
# ============================================

def prepare_data_for_modeling(df, numerical_features, categorical_features, target_col):
    """准备建模数据"""
    print("\n" + "="*50)
    print("第三步：数据预处理")
    print("="*50)
    
    X = df[numerical_features + categorical_features].copy()
    y_original = df[target_col].copy()
    
    # 标签编码：将1,2,3转换为0,1,2
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_original)
    
    print(f"标签映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # 标准化数值特征
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # 独热编码
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    
    print(f"编码后特征数量: {X_encoded.shape[1]}")
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集分布: {np.bincount(y_train)}")
    print(f"测试集分布: {np.bincount(y_test)}")
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"类别权重: {class_weight_dict}")
    
    return X_train, X_test, y_train, y_test, scaler, X_encoded.columns, class_weight_dict, label_encoder

# ============================================
# 第四部分：训练模型
# ============================================

def get_models(class_weight_dict):
    """定义六个机器学习模型"""
    class_weights_list = [class_weight_dict.get(i, 1.0) for i in range(3)]
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, 
            class_weight=class_weight_dict
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, 
            class_weight=class_weight_dict, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            objective='multi:softprob', random_state=42, 
            eval_metric='mlogloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, num_leaves=31, learning_rate=0.1,
            objective='multiclass', random_state=42, verbose=-1,
            class_weight=class_weight_dict
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            loss_function='MultiClass', random_seed=42, verbose=0,
            class_weights=class_weights_list
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50), activation='relu', 
            solver='adam', max_iter=500, random_state=42, early_stopping=True
        )
    }
    return models

def train_all_models(X_train, y_train, X_test, y_test, class_weight_dict, label_encoder):
    """训练所有模型"""
    print("\n" + "="*50)
    print("第四步：模型训练与评估")
    print("="*50)
    
    models = get_models(class_weight_dict)
    results = {}
    trained_models = {}
    
    target_names = ['Type1', 'Type2', 'Type3']
    
    for name, model in models.items():
        print(f"\n--- 训练 {name} ---")
        
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  [OK] {name} 训练成功")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # 计算评估指标
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
            results[name] = {
                'Accuracy': report['accuracy'],
                'Macro Precision': report['macro avg']['precision'],
                'Macro Recall': report['macro avg']['recall'],
                'Macro F1': report['macro avg']['f1-score'],
            }
            
            # 打印分类报告
            print(f"  Accuracy: {report['accuracy']:.4f}")
            print(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
            
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"  [ERROR] {name} 训练失败: {e}")
    
    # 模型对比表
    results_df = pd.DataFrame(results).T.round(4)
    
    print("\n" + "="*60)
    print("模型对比汇总表")
    print("="*60)
    print(results_df[['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']].to_string())
    
    if len(results_df) > 0:
        best_model_name = results_df['Macro F1'].idxmax()
        print(f"\n最佳模型: {best_model_name}")
    else:
        best_model_name = None
    
    try:
        results_df.to_excel('model_comparison.xlsx')
    except:
        results_df.to_csv('model_comparison.csv')
    
    return results_df, trained_models, best_model_name

# ============================================
# 第五部分：模型性能对比图
# ============================================

def plot_model_comparison(results_df):
    """绘制模型性能对比图"""
    print("\n" + "="*50)
    print("第五步：模型性能对比图")
    print("="*50)
    
    if len(results_df) == 0:
        print("没有成功训练的模型")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = results_df.index
    x = np.arange(len(models))
    width = 0.25
    
    accuracy = results_df['Accuracy'].values
    macro_f1 = results_df['Macro F1'].values
    macro_recall = results_df['Macro Recall'].values
    
    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, macro_f1, width, label='Macro F1', color='lightcoral', alpha=0.8)
    bars3 = ax.bar(x + width, macro_recall, width, label='Macro Recall', color='lightgreen', alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    plt.show()
    print("模型性能对比图已保存")

# ============================================
# 第六部分：ROC曲线（每个模型单独绘制，确保有线）
# ============================================

def plot_roc_curves_separate(models, X_test, y_test):
    """为每个模型单独绘制ROC曲线（确保有线）"""
    print("\n" + "="*50)
    print("第六步：ROC曲线分析（每个模型单独绘制）")
    print("="*50)
    
    n_classes = 3
    class_names = ['Type 1', 'Type 2', 'Type 3']
    colors = ['blue', 'red', 'green']
    
    macro_aucs = {}
    
    for model_name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)
            
            # 创建新图形
            plt.figure(figsize=(10, 8))
            
            auc_scores = []
            for i in range(n_classes):
                y_test_binary = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores.append(roc_auc)
                
                plt.plot(fpr, tpr, lw=2, color=colors[i], 
                        label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
            
            macro_auc = np.mean(auc_scores)
            macro_aucs[model_name] = macro_auc
            
            plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curves - {model_name}\nMacro AUC = {macro_auc:.3f}', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'roc_curves_{model_name.replace(" ", "_")}.png', dpi=300)
            plt.show()
            
            print(f"\n{model_name} - 各类别AUC: {[f'{x:.3f}' for x in auc_scores]}, Macro AUC: {macro_auc:.3f}")
            
        except Exception as e:
            print(f"绘制 {model_name} 的ROC曲线时出错: {e}")
    
    # 绘制所有模型对比图（使用宏平均AUC）
    if macro_aucs:
        plt.figure(figsize=(10, 6))
        model_names = list(macro_aucs.keys())
        auc_values = list(macro_aucs.values())
        
        bars = plt.bar(model_names, auc_values, color='steelblue', alpha=0.8)
        plt.ylim([0.5, 1.0])
        plt.ylabel('Macro Average AUC', fontsize=12)
        plt.title('Model Comparison - Macro AUC', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, auc_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('macro_auc_comparison.png', dpi=300)
        plt.show()
        
        print("\n宏平均AUC结果:")
        for name, auc_val in macro_aucs.items():
            print(f"  {name}: {auc_val:.4f}")
    
    return macro_aucs

# ============================================
# 第七部分：DCA曲线
# ============================================

def calculate_net_benefit(y_true, y_pred_proba, thresholds):
    """计算净获益"""
    n = len(y_true)
    net_benefit = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        if threshold < 1:
            nb = tp / n - (fp / n) * (threshold / (1 - threshold))
        else:
            nb = tp / n
        net_benefit.append(nb)
    
    return np.array(net_benefit)

def plot_dca_curves_all_models(models, X_test, y_test):
    """绘制所有模型的DCA曲线在一张图上"""
    print("\n" + "="*50)
    print("第七步：决策曲线分析 (DCA)")
    print("="*50)
    
    thresholds = np.linspace(0.01, 0.99, 30)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    class_names = ['Type 1', 'Type 2', 'Type 3']
    
    for class_idx in range(3):
        plt.figure(figsize=(10, 8))
        
        color_idx = 0
        for model_name, model in models.items():
            try:
                y_pred_proba = model.predict_proba(X_test)
                y_true_binary = (y_test == class_idx).astype(int)
                y_prob_binary = y_pred_proba[:, class_idx]
                
                nb_model = calculate_net_benefit(y_true_binary, y_prob_binary, thresholds)
                plt.plot(thresholds, nb_model, lw=2, 
                        label=f'{model_name}', color=colors[color_idx % len(colors)])
                color_idx += 1
            except Exception as e:
                print(f"绘制 {model_name} 的DCA曲线时出错: {e}")
        
        # Treat All和Treat None曲线
        y_true_binary = (y_test == class_idx).astype(int)
        prevalence = y_true_binary.mean()
        nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        nb_treat_none = np.zeros_like(thresholds)
        
        plt.plot(thresholds, nb_treat_all, 'k--', lw=2, label='Treat All')
        plt.plot(thresholds, nb_treat_none, 'k:', lw=2, label='Treat None')
        
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title(f'Decision Curve Analysis - {class_names[class_idx]}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.savefig(f'dca_curves_all_models_{class_names[class_idx].replace(" ", "_")}.png', dpi=300)
        plt.show()
    
    print("DCA分析完成")

# ============================================
# 第八部分：SHAP分析
# ============================================

def shap_analysis(best_model, X_test, feature_names, best_model_name):
    """进行SHAP分析"""
    print("\n" + "="*50)
    print("第八步：SHAP分析")
    print("="*50)
    
    try:
        import shap
        
        X_sample = X_test
        
        if best_model_name.lower() in ['xgboost', 'random forest', 'lightgbm', 'catboost']:
            print("正在计算SHAP值...")
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_sample)
            
            # 特征重要性图（所有特征）
            plt.figure(figsize=(14, 10))
            
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean(np.abs(shap_values), axis=1).mean(axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            feature_importance = pd.DataFrame({
                'feature': feature_names[:len(mean_abs_shap)],
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            plt.barh(range(len(feature_importance)), feature_importance['importance'].values)
            plt.yticks(range(len(feature_importance)), feature_importance['feature'].values)
            plt.xlabel('Mean |SHAP value|', fontsize=12)
            plt.title(f'SHAP Feature Importance - All {len(feature_importance)} Features\n{best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('shap_feature_importance_all.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            try:
                feature_importance.to_excel('shap_feature_importance_all.xlsx', index=False)
                print("特征重要性已保存")
            except:
                feature_importance.to_csv('shap_feature_importance_all.csv', index=False)
            
            print(f"\n前10个最重要的特征:")
            print(feature_importance.head(10).to_string())
            
            print("SHAP分析完成")
        else:
            print(f"模型 {best_model_name} 不支持TreeExplainer")
            
    except ImportError:
        print("shap库未安装，跳过SHAP分析")
        print("如需SHAP分析，请运行: pip install shap")
    except Exception as e:
        print(f"SHAP分析出错: {e}")

# ============================================
# 第九部分：最终报告
# ============================================

def generate_final_report(best_model_name, results_df, macro_aucs):
    """生成最终报告"""
    print("\n" + "="*50)
    print("第九步：最终分析报告")
    print("="*50)
    
    report = f"""
    ========================================
    三分类癌症预测模型分析报告
    ========================================
    
    1. 最佳模型: {best_model_name}
    
    2. 模型性能对比
    ---------------
    """
    
    for model_name in results_df.index:
        report += f"\n   {model_name}:"
        report += f"\n     Accuracy: {results_df.loc[model_name, 'Accuracy']:.4f}"
        report += f"\n     Macro F1: {results_df.loc[model_name, 'Macro F1']:.4f}"
        if macro_aucs and model_name in macro_aucs:
            report += f"\n     Macro AUC: {macro_aucs[model_name]:.4f}"
    
    report += """
    
    3. 输出文件清单
    ---------------
    描述性统计:
    - boxplot_*.png: 每个数值变量的箱线图
    - barplot_*.png: 每个分类变量的柱状图
    
    模型评估:
    - model_comparison.xlsx/csv: 模型性能对比表
    - confusion_matrix_*.png: 各模型的混淆矩阵
    - model_performance_comparison.png: 模型性能对比图
    
    ROC曲线:
    - roc_curves_*.png: 每个模型的ROC曲线（单独绘制）
    - macro_auc_comparison.png: 宏平均AUC对比图
    
    DCA曲线:
    - dca_curves_all_models_*.png: DCA曲线
    
    SHAP分析:
    - shap_feature_importance_all.png: 特征重要性图
    - shap_feature_importance_all.xlsx/csv: 特征重要性数据
    """
    
    print(report)
    
    try:
        with open('final_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("报告已保存")
    except:
        with open('final_report.txt', 'w') as f:
            f.write(report)

# ============================================
# 主函数
# ============================================

def main():
    print("="*60)
    print("三分类癌症预测模型完整分析流程")
    print("="*60)
    
    file_path = "D:\临时文件\文档\文献\三分类癌症预测模型分析\lpa\data.xlsx"
    
    # 1. 加载数据
    df, numerical_features, categorical_features, target_col = load_and_preprocess_data(file_path)
    if df is None:
        return
    
    # 2. 描述性统计分析
    descriptive_statistics(df, numerical_features, categorical_features, target_col)
    
    # 3. 准备建模数据
    X_train, X_test, y_train, y_test, scaler, feature_names, class_weight_dict, label_encoder = prepare_data_for_modeling(
        df, numerical_features, categorical_features, target_col
    )
    
    # 4. 训练所有模型
    results_df, trained_models, best_model_name = train_all_models(
        X_train, y_train, X_test, y_test, class_weight_dict, label_encoder
    )
    
    if len(trained_models) == 0:
        print("没有成功训练的模型")
        return
    
    # 5. 模型性能对比图
    plot_model_comparison(results_df)
    
    # 6. ROC曲线（每个模型单独绘制，确保有线）
    macro_aucs = plot_roc_curves_separate(trained_models, X_test, y_test)
    
    # 7. DCA曲线
    plot_dca_curves_all_models(trained_models, X_test, y_test)
    
    # 8. SHAP分析
    if best_model_name and best_model_name in trained_models:
        best_model = trained_models[best_model_name]
        shap_analysis(best_model, X_test, feature_names, best_model_name)
    
    # 9. 生成报告
    generate_final_report(best_model_name, results_df, macro_aucs)
    
    # 10. 保存模型
    if best_model_name and best_model_name in trained_models:
        try:
            joblib.dump(trained_models[best_model_name], f'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(label_encoder, 'label_encoder.pkl')
            print("\n模型已保存")
        except Exception as e:
            print(f"保存失败: {e}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()