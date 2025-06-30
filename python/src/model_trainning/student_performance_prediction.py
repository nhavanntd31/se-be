import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
import argparse
import pickle
import time
warnings.filterwarnings('ignore')

class StudentPerformanceDataProcessor:
    def __init__(self):
        self.grade_mapping = {
            'A+': 4.0, 'A': 4, 'B+': 3.5, 'B': 3, 'C+': 2.5, 'C': 2,
            'D+': 1.5, 'D': 1, 'F': 0, 'X': 0, 'W': 0
        }
        self.warning_mapping = {
            'Mức 0': 0, 'Mức 1': 1, 'Mức 2': 2, 'Mức 3': 3
        }
        self.course_scaler = StandardScaler()
        self.performance_scaler = StandardScaler()
        
    def clean_numeric_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            df[col] = df[col].replace([np.inf, -np.inf], mean_val)
        return df
        
    def preprocess_course_data(self, course_df: pd.DataFrame) -> pd.DataFrame:
        df = course_df.copy()
        
        df['Final Grade Numeric'] = df['Final Grade'].map(self.grade_mapping)
        
        numeric_features = ['Continuous Assessment Score', 'Exam Score', 'Credits', 
                          'Final Grade Numeric']
        
        df = self.clean_numeric_data(df, numeric_features)
        
        df['Course_Category'] = df['Course ID'].str[:2]
        course_cat_encoder = LabelEncoder()
        df['Course_Category_Encoded'] = course_cat_encoder.fit_transform(df['Course_Category'])
        
        df['Pass_Status'] = (df['Final Grade Numeric'] >= 1.0).astype(int)
        df['Grade_Points'] = df['Final Grade Numeric'] * df['Credits']
        
        all_features = numeric_features + ['Course_Category_Encoded', 'Pass_Status', 'Grade_Points']
        
        df[all_features] = self.course_scaler.fit_transform(df[all_features])
        
        return df[['Semester', 'student_id', 'Relative Term'] + all_features]
    
    def preprocess_performance_data(self, perf_df: pd.DataFrame) -> pd.DataFrame:
        df = perf_df.copy()
        
        numeric_cols = ['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg']
        df = self.clean_numeric_data(df, numeric_cols)
        
        df['Warning_Numeric'] = df['Warning'].map(self.warning_mapping).fillna(0)
        
        df['Level_Year'] = df['Level'].str.extract('(\d+)').astype(float).fillna(1)
        
        df['Pass_Rate'] = df['TC qua'] / (df['Reg'] + 1e-8)
        df['Debt_Rate'] = df['Debt'] / (df['Reg'] + 1e-8)
        df['Accumulation_Rate'] = df['Acc'] / (df['Relative Term'] * 20 + 1e-8)
        
        rate_cols = ['Pass_Rate', 'Debt_Rate', 'Accumulation_Rate']
        df = self.clean_numeric_data(df, rate_cols)
        
        performance_features = ['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                              'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                              'Debt_Rate', 'Accumulation_Rate']
        
        df[performance_features] = self.performance_scaler.fit_transform(df[performance_features])
        
        return df[['Semester', 'student_id', 'Relative Term'] + performance_features]

class StudentSequenceDataset(Dataset):
    def __init__(self, student_sequences: List[Dict], max_courses_per_semester: int = 15, 
                 max_semesters: int = 10):
        self.sequences = student_sequences
        self.max_courses = max_courses_per_semester
        self.max_semesters = max_semesters
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        course_features = torch.zeros(self.max_semesters, self.max_courses, 7)
        semester_features = torch.zeros(self.max_semesters, 11)
        course_masks = torch.zeros(self.max_semesters, self.max_courses)
        semester_masks = torch.zeros(self.max_semesters)
        
        for sem_idx, sem_data in enumerate(seq['semesters'][:self.max_semesters]):
            semester_features[sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
            semester_masks[sem_idx] = 1.0
            
            courses = sem_data['courses'][:self.max_courses]
            for course_idx, course in enumerate(courses):
                course_features[sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
                course_masks[sem_idx, course_idx] = 1.0
        
        target_gpa = seq['target_gpa']
        target_cpa = seq['target_cpa']
        
        return {
            'course_features': course_features,
            'semester_features': semester_features, 
            'course_masks': course_masks,
            'semester_masks': semester_masks,
            'target_gpa': torch.tensor(target_gpa, dtype=torch.float32),
            'target_cpa': torch.tensor(target_cpa, dtype=torch.float32)
        }

class CourseEncoder(nn.Module):
    def __init__(self, course_feature_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.course_embedding = nn.Sequential(
            nn.Linear(course_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, course_features, course_masks):
        batch_size, max_sems, max_courses, feat_dim = course_features.shape
        
        course_features_flat = course_features.view(-1, feat_dim)
        course_embeddings = self.course_embedding(course_features_flat)
        course_embeddings = course_embeddings.view(batch_size, max_sems, max_courses, -1)
        
        masked_embeddings = course_embeddings * course_masks.unsqueeze(-1)
        semester_course_sum = masked_embeddings.sum(dim=2)
        
        course_counts = course_masks.sum(dim=2, keepdim=True)
        semester_representations = semester_course_sum / (course_counts + 1e-8)
        
        return semester_representations

class SemesterTransformerEncoder(nn.Module):
    def __init__(self, semester_dim: int = 64, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(50, semester_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=semester_dim,
            nhead=num_heads,
            dim_feedforward=semester_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, semester_representations, semester_masks):
        seq_len = semester_representations.size(1)
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
        
        enhanced_representations = semester_representations + pos_enc
        
        attention_mask = ~semester_masks.bool()
        
        transformed = self.transformer(enhanced_representations, src_key_padding_mask=attention_mask)
        
        return transformed

class StudentPerformancePredictor(nn.Module):
    def __init__(self, course_feature_dim: int = 7, semester_feature_dim: int = 11, 
                 hidden_dim: int = 64, lstm_hidden: int = 128, num_heads: int = 8):
        super().__init__()
        
        self.course_encoder = CourseEncoder(course_feature_dim, hidden_dim)
        
        self.semester_feature_proj = nn.Linear(semester_feature_dim, hidden_dim)
        
        self.semester_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.transformer_encoder = SemesterTransformerEncoder(hidden_dim, num_heads, 2)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, course_features, semester_features, course_masks, semester_masks):
        course_representations = self.course_encoder(course_features, course_masks)
        
        semester_proj = self.semester_feature_proj(semester_features)
        
        fused_representations = self.semester_fusion(
            torch.cat([course_representations, semester_proj], dim=-1)
        )
        
        transformer_output = self.transformer_encoder(fused_representations, semester_masks)
        
        lstm_output, _ = self.lstm(transformer_output)
        
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output,
                                       key_padding_mask=~semester_masks.bool())
        
        valid_lengths = semester_masks.sum(dim=1).long()
        batch_indices = torch.arange(attn_output.size(0))
        last_valid_outputs = attn_output[batch_indices, valid_lengths - 1]
        
        predictions = self.predictor(last_valid_outputs)
        
        return predictions

class ModelTrainer:
    def __init__(self, model: nn.Module, processor: StudentPerformanceDataProcessor = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.processor = processor
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, min_lr=1e-6)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            course_features = batch['course_features'].to(self.device)
            semester_features = batch['semester_features'].to(self.device)
            course_masks = batch['course_masks'].to(self.device)
            semester_masks = batch['semester_masks'].to(self.device)
            target_gpa = batch['target_gpa'].to(self.device)
            target_cpa = batch['target_cpa'].to(self.device)
            
            targets = torch.stack([target_gpa, target_cpa], dim=1)
            
            self.optimizer.zero_grad()
            predictions = self.model(course_features, semester_features, course_masks, semester_masks)
            
            if torch.isnan(predictions).any():
                print("Warning: NaN detected in predictions")
                continue
                
            loss = self.criterion(predictions, targets)
            
            if torch.isnan(loss):
                print("Warning: NaN detected in loss")
                continue
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                course_features = batch['course_features'].to(self.device)
                semester_features = batch['semester_features'].to(self.device)
                course_masks = batch['course_masks'].to(self.device)
                semester_masks = batch['semester_masks'].to(self.device)
                target_gpa = batch['target_gpa'].to(self.device)
                target_cpa = batch['target_cpa'].to(self.device)
                
                targets = torch.stack([target_gpa, target_cpa], dim=1)
                
                predictions = self.model(course_features, semester_features, course_masks, semester_masks)
                
                if torch.isnan(predictions).any():
                    continue
                    
                loss = self.criterion(predictions, targets)
                
                if torch.isnan(loss):
                    continue
                    
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            return {
                'loss': float('inf'),
                'gpa_mse': float('inf'), 'cpa_mse': float('inf'),
                'gpa_mae': float('inf'), 'cpa_mae': float('inf'),
                'gpa_r2': float('-inf'), 'cpa_r2': float('-inf')
            }
            
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        valid_mask = ~np.isnan(all_predictions).any(axis=1) & ~np.isnan(all_targets).any(axis=1)
        valid_predictions = all_predictions[valid_mask]
        valid_targets = all_targets[valid_mask]
        
        if len(valid_predictions) == 0:
            return {
                'loss': total_loss / len(dataloader),
                'gpa_mse': float('inf'), 'cpa_mse': float('inf'),
                'gpa_mae': float('inf'), 'cpa_mae': float('inf'),
                'gpa_r2': float('-inf'), 'cpa_r2': float('-inf')
            }
        
        gpa_mse = mean_squared_error(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_mse = mean_squared_error(valid_targets[:, 1], valid_predictions[:, 1])
        gpa_mae = mean_absolute_error(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_mae = mean_absolute_error(valid_targets[:, 1], valid_predictions[:, 1])
        gpa_r2 = r2_score(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_r2 = r2_score(valid_targets[:, 1], valid_predictions[:, 1])
        
        results = {
            'loss': total_loss / len(dataloader),
            'gpa_mse': gpa_mse, 'cpa_mse': cpa_mse,
            'gpa_mae': gpa_mae, 'cpa_mae': cpa_mae,
            'gpa_r2': gpa_r2, 'cpa_r2': cpa_r2
        }
        
        if self.processor and hasattr(self.processor, 'performance_scaler'):
            gpa_col_idx = 0
            cpa_col_idx = 1
            
            dummy_features = np.zeros((len(valid_predictions), 11))
            dummy_features[:, gpa_col_idx] = valid_predictions[:, 0]
            dummy_features[:, cpa_col_idx] = valid_predictions[:, 1]
            
            dummy_targets = np.zeros((len(valid_targets), 11))
            dummy_targets[:, gpa_col_idx] = valid_targets[:, 0]
            dummy_targets[:, cpa_col_idx] = valid_targets[:, 1]
            
            try:
                denorm_predictions = self.processor.performance_scaler.inverse_transform(dummy_features)
                denorm_targets = self.processor.performance_scaler.inverse_transform(dummy_targets)
                
                gpa_pred_denorm = denorm_predictions[:, gpa_col_idx]
                gpa_target_denorm = denorm_targets[:, gpa_col_idx]
                cpa_pred_denorm = denorm_predictions[:, cpa_col_idx]
                cpa_target_denorm = denorm_targets[:, cpa_col_idx]
                
                gpa_mse_denorm = mean_squared_error(gpa_target_denorm, gpa_pred_denorm)
                cpa_mse_denorm = mean_squared_error(cpa_target_denorm, cpa_pred_denorm)
                gpa_mae_denorm = mean_absolute_error(gpa_target_denorm, gpa_pred_denorm)
                cpa_mae_denorm = mean_absolute_error(cpa_target_denorm, cpa_pred_denorm)
                gpa_r2_denorm = r2_score(gpa_target_denorm, gpa_pred_denorm)
                cpa_r2_denorm = r2_score(cpa_target_denorm, cpa_pred_denorm)
                
                results.update({
                    'gpa_mse_denorm': gpa_mse_denorm,
                    'cpa_mse_denorm': cpa_mse_denorm,
                    'gpa_mae_denorm': gpa_mae_denorm,
                    'cpa_mae_denorm': cpa_mae_denorm,
                    'gpa_r2_denorm': gpa_r2_denorm,
                    'cpa_r2_denorm': cpa_r2_denorm
                })
            except Exception as e:
                print(f"Warning: Could not compute denormalized metrics: {e}")
        
        return results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        train_losses = []
        val_losses = []
        gpa_mse_history = []
        gpa_mae_history = []
        cpa_mse_history = []
        cpa_mae_history = []
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            gpa_mse_history.append(val_metrics['gpa_mse'])
            gpa_mae_history.append(val_metrics['gpa_mae'])
            cpa_mse_history.append(val_metrics['cpa_mse'])
            cpa_mae_history.append(val_metrics['cpa_mae'])
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
            print(f'  GPA - MSE: {val_metrics["gpa_mse"]:.4f}, MAE: {val_metrics["gpa_mae"]:.4f}, R2: {val_metrics["gpa_r2"]:.4f}')
            print(f'  CPA - MSE: {val_metrics["cpa_mse"]:.4f}, MAE: {val_metrics["cpa_mae"]:.4f}, R2: {val_metrics["cpa_r2"]:.4f}')
            
            if 'gpa_mse_denorm' in val_metrics:
                print(f'  [DENORM] GPA - MSE: {val_metrics["gpa_mse_denorm"]:.4f}, MAE: {val_metrics["gpa_mae_denorm"]:.4f}, R2: {val_metrics["gpa_r2_denorm"]:.4f}')
                print(f'  [DENORM] CPA - MSE: {val_metrics["cpa_mse_denorm"]:.4f}, MAE: {val_metrics["cpa_mae_denorm"]:.4f}, R2: {val_metrics["cpa_r2_denorm"]:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'gpa_mse': gpa_mse_history,
            'gpa_mae': gpa_mae_history,
            'cpa_mse': cpa_mse_history,
            'cpa_mae': cpa_mae_history
        }

def is_active_semester(perf_row, course_count):
    gpa = perf_row['GPA'] if 'GPA' in perf_row else 0
    tc_qua = perf_row['TC qua'] if 'TC qua' in perf_row else 0
    
    if pd.isna(gpa) or gpa == 0:
        return False
    if course_count == 0:
        return False
    if pd.isna(tc_qua) or tc_qua == 0:
        return False
    
    return True

def create_student_sequences(course_df: pd.DataFrame, perf_df: pd.DataFrame, 
                           processor: StudentPerformanceDataProcessor) -> List[Dict]:
    course_processed = processor.preprocess_course_data(course_df)
    perf_processed = processor.preprocess_performance_data(perf_df)
    sequences = []
    skipped_students = []
    dropped_out_students = []
    
    for student_id in course_processed['student_id'].unique():
        student_courses = course_processed[course_processed['student_id'] == student_id]
        student_perf = perf_processed[perf_processed['student_id'] == student_id]
        
        if len(student_perf) < 2:
            skipped_students.append(f"{student_id} (ít hơn 2 kỳ)")
            continue
        
        semester_data = []
        has_null = False
        
        for _, perf_row in student_perf.iterrows():
            semester = perf_row['Semester']
            semester_courses = student_courses[student_courses['Semester'] == semester]
            
            if not is_active_semester(perf_row, len(semester_courses)):
                print(f"Sinh viên {student_id} đã bỏ học/nghỉ học từ kỳ {semester}")
                break
            
            if len(semester_courses) == 0:
                continue
                
            course_features = semester_courses[['Continuous Assessment Score', 'Exam Score', 'Credits',
                                             'Final Grade Numeric', 'Course_Category_Encoded', 
                                             'Pass_Status', 'Grade_Points']].values.tolist()
            
            performance_features = perf_row[['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                                           'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                                           'Debt_Rate', 'Accumulation_Rate']].values.tolist()
            
            if np.isnan(course_features).any() or np.isnan(performance_features).any():
                has_null = True
                break
                
            semester_data.append({
                'semester': semester,
                'relative_term': perf_row['Relative Term'],
                'courses': course_features,
                'performance': performance_features,
                'gpa': perf_row['GPA'] if 'GPA' in perf_row else 0,
                'tc_qua': perf_row['TC qua'] if 'TC qua' in perf_row else 0
            })
        
        if has_null:
            skipped_students.append(f"{student_id} (dữ liệu null)")
            continue
            
        if len(semester_data) < 2:
            dropped_out_students.append(f"{student_id} (bỏ học sớm)")
            continue
            
        semester_data.sort(key=lambda x: x['relative_term'])
        
        valid_sequences_count = 0
        for i in range(len(semester_data) - 1):
            input_semesters = semester_data[:i+1]
            target_semester = semester_data[i+1]
            
            target_gpa = target_semester['gpa']
            target_tc = target_semester['tc_qua']
            
            if target_gpa > 0 and target_tc > 0:
                sequences.append({
                    'student_id': student_id,
                    'semesters': input_semesters,
                    'target_gpa': target_semester['performance'][0],
                    'target_cpa': target_semester['performance'][1]
                })
                valid_sequences_count += 1
        
        if valid_sequences_count == 0:
            dropped_out_students.append(f"{student_id} (không có sequence hợp lệ)")
    
    print(f"\nSố sinh viên bị loại do dữ liệu null/thiếu: {len(skipped_students)}")
    print(f"Số sinh viên bỏ học/nghỉ học: {len(dropped_out_students)}")
    print(f"Danh sách sinh viên bỏ học: {dropped_out_students[:10]}...")
    
    return sequences

class TraditionalMLTrainer:
    def __init__(self, processor: StudentPerformanceDataProcessor):
        self.processor = processor
        self.models = {}
        self.trained_models = {}
        
    def prepare_ml_data(self, sequences: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        targets = []
        
        for seq in sequences:
            student_features = []
            
            for sem_data in seq['semesters']:
                course_features = np.array(sem_data['courses'])
                if len(course_features) > 0:
                    course_stats = [
                        np.mean(course_features, axis=0),
                        np.std(course_features, axis=0),
                        np.min(course_features, axis=0),
                        np.max(course_features, axis=0),
                        [len(course_features)]
                    ]
                    course_aggregated = np.concatenate([stat.flatten() for stat in course_stats])
                else:
                    course_aggregated = np.zeros(29)
                
                performance_features = np.array(sem_data['performance'])
                combined_features = np.concatenate([course_aggregated, performance_features])
                student_features.extend(combined_features)
            
            if len(student_features) > 0:
                max_length = 400
                if len(student_features) > max_length:
                    student_features = student_features[:max_length]
                else:
                    student_features.extend([0] * (max_length - len(student_features)))
                
                features.append(student_features)
                targets.append([seq['target_gpa'], seq['target_cpa']])
        
        return np.array(features), np.array(targets)
    
    def initialize_models(self):
        self.models = {
            'Random Forest': MultiOutputRegressor(RandomForestRegressor(
                n_estimators=100, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )),
            'Linear Regression': MultiOutputRegressor(LinearRegression()),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0, random_state=42)),
            'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.1, random_state=42)),
            'SVM': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale'))
        }
    
    def evaluate_model_denormalized(self, model, X_test, y_test_normalized):
        y_pred_normalized = model.predict(X_test)
        
        normalized_metrics = {
            'gpa_mse': mean_squared_error(y_test_normalized[:, 0], y_pred_normalized[:, 0]),
            'cpa_mse': mean_squared_error(y_test_normalized[:, 1], y_pred_normalized[:, 1]),
            'gpa_mae': mean_absolute_error(y_test_normalized[:, 0], y_pred_normalized[:, 0]),
            'cpa_mae': mean_absolute_error(y_test_normalized[:, 1], y_pred_normalized[:, 1]),
            'gpa_r2': r2_score(y_test_normalized[:, 0], y_pred_normalized[:, 0]),
            'cpa_r2': r2_score(y_test_normalized[:, 1], y_pred_normalized[:, 1])
        }
        
        dummy_targets = np.zeros((len(y_test_normalized), 11))
        dummy_targets[:, 0] = y_test_normalized[:, 0]
        dummy_targets[:, 1] = y_test_normalized[:, 1]
        
        dummy_predictions = np.zeros((len(y_pred_normalized), 11))
        dummy_predictions[:, 0] = y_pred_normalized[:, 0]
        dummy_predictions[:, 1] = y_pred_normalized[:, 1]
        
        try:
            y_test_denorm = self.processor.performance_scaler.inverse_transform(dummy_targets)
            y_pred_denorm = self.processor.performance_scaler.inverse_transform(dummy_predictions)
            
            denormalized_metrics = {
                'gpa_mse_denorm': mean_squared_error(y_test_denorm[:, 0], y_pred_denorm[:, 0]),
                'cpa_mse_denorm': mean_squared_error(y_test_denorm[:, 1], y_pred_denorm[:, 1]),
                'gpa_mae_denorm': mean_absolute_error(y_test_denorm[:, 0], y_pred_denorm[:, 0]),
                'cpa_mae_denorm': mean_absolute_error(y_test_denorm[:, 1], y_pred_denorm[:, 1]),
                'gpa_r2_denorm': r2_score(y_test_denorm[:, 0], y_pred_denorm[:, 0]),
                'cpa_r2_denorm': r2_score(y_test_denorm[:, 1], y_pred_denorm[:, 1])
            }
        except Exception as e:
            print(f"Warning: Could not compute denormalized metrics: {e}")
            denormalized_metrics = {
                'gpa_mse_denorm': float('inf'),
                'cpa_mse_denorm': float('inf'),
                'gpa_mae_denorm': float('inf'),
                'cpa_mae_denorm': float('inf'),
                'gpa_r2_denorm': float('-inf'),
                'cpa_r2_denorm': float('-inf')
            }
        
        return {**normalized_metrics, **denormalized_metrics}
    
    def train_and_evaluate_all(self, train_sequences, test_sequences):
        print("\n" + "="*60)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*60)
        
        X_train, y_train = self.prepare_ml_data(train_sequences)
        X_test, y_test = self.prepare_ml_data(test_sequences)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        self.initialize_models()
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                metrics = self.evaluate_model_denormalized(model, X_test, y_test)
                metrics['training_time'] = training_time
                
                results[model_name] = metrics
                self.trained_models[model_name] = model
                
                print(f"  Training time: {training_time:.2f}s")
                print(f"  GPA (Normalized) - MSE: {metrics['gpa_mse']:.4f}, MAE: {metrics['gpa_mae']:.4f}, R²: {metrics['gpa_r2']:.4f}")
                print(f"  CPA (Normalized) - MSE: {metrics['cpa_mse']:.4f}, MAE: {metrics['cpa_mae']:.4f}, R²: {metrics['cpa_r2']:.4f}")
                
                if 'gpa_mse_denorm' in metrics and metrics['gpa_mse_denorm'] != float('inf'):
                    print(f"  GPA (Denormalized) - MSE: {metrics['gpa_mse_denorm']:.4f}, MAE: {metrics['gpa_mae_denorm']:.4f}, R²: {metrics['gpa_r2_denorm']:.4f}")
                    print(f"  CPA (Denormalized) - MSE: {metrics['cpa_mse_denorm']:.4f}, MAE: {metrics['cpa_mae_denorm']:.4f}, R²: {metrics['cpa_r2_denorm']:.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def save_best_models(self, results):
        best_gpa_model = None
        best_cpa_model = None
        best_gpa_mae = float('inf')
        best_cpa_mae = float('inf')
        
        for model_name, metrics in results.items():
            if metrics is None:
                continue
                
            if 'gpa_mae_denorm' in metrics and metrics['gpa_mae_denorm'] < best_gpa_mae:
                best_gpa_mae = metrics['gpa_mae_denorm']
                best_gpa_model = (model_name, self.trained_models[model_name])
            
            if 'cpa_mae_denorm' in metrics and metrics['cpa_mae_denorm'] < best_cpa_mae:
                best_cpa_mae = metrics['cpa_mae_denorm']
                best_cpa_model = (model_name, self.trained_models[model_name])
        
        if best_gpa_model:
            with open('best_gpa_model.pkl', 'wb') as f:
                pickle.dump(best_gpa_model[1], f)
            print(f"\nBest GPA model ({best_gpa_model[0]}) saved to 'best_gpa_model.pkl'")
            print(f"GPA MAE (denormalized): {best_gpa_mae:.4f}")
        
        if best_cpa_model:
            with open('best_cpa_model.pkl', 'wb') as f:
                pickle.dump(best_cpa_model[1], f)
            print(f"Best CPA model ({best_cpa_model[0]}) saved to 'best_cpa_model.pkl'")
            print(f"CPA MAE (denormalized): {best_cpa_mae:.4f}")
        
        return best_gpa_model, best_cpa_model

def plot_ml_comparison(ml_results, dl_results=None):
    models = list(ml_results.keys())
    models = [m for m in models if ml_results[m] is not None]
    
    gpa_mae_denorm = [ml_results[m]['gpa_mae_denorm'] for m in models if 'gpa_mae_denorm' in ml_results[m]]
    cpa_mae_denorm = [ml_results[m]['cpa_mae_denorm'] for m in models if 'cpa_mae_denorm' in ml_results[m]]
    gpa_mse_denorm = [ml_results[m]['gpa_mse_denorm'] for m in models if 'gpa_mse_denorm' in ml_results[m]]
    cpa_mse_denorm = [ml_results[m]['cpa_mse_denorm'] for m in models if 'cpa_mse_denorm' in ml_results[m]]
    
    if dl_results and 'gpa_mae_denorm' in dl_results:
        models.append('Deep Learning')
        gpa_mae_denorm.append(dl_results['gpa_mae_denorm'])
        cpa_mae_denorm.append(dl_results['cpa_mae_denorm'])
        gpa_mse_denorm.append(dl_results['gpa_mse_denorm'])
        cpa_mse_denorm.append(dl_results['cpa_mse_denorm'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars1 = ax1.bar(models, gpa_mae_denorm, color=colors, alpha=0.7)
    ax1.set_title('GPA Mean Absolute Error (Denormalized)')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, gpa_mae_denorm):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gpa_mae_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    bars2 = ax2.bar(models, cpa_mae_denorm, color=colors, alpha=0.7)
    ax2.set_title('CPA Mean Absolute Error (Denormalized)')
    ax2.set_ylabel('MAE')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, cpa_mae_denorm):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpa_mae_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    bars3 = ax3.bar(models, gpa_mse_denorm, color=colors, alpha=0.7)
    ax3.set_title('GPA Mean Squared Error (Denormalized)')
    ax3.set_ylabel('MSE')
    ax3.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars3, gpa_mse_denorm):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gpa_mse_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    bars4 = ax4.bar(models, cpa_mse_denorm, color=colors, alpha=0.7)
    ax4.set_title('CPA Mean Squared Error (Denormalized)')
    ax4.set_ylabel('MSE')
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars4, cpa_mse_denorm):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpa_mse_denorm)*0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ml_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSo sánh các mô hình đã được lưu vào 'ml_algorithms_comparison.png'")

def plot_detailed_training_metrics(training_history, test_metrics, base_path='detailed_metrics'):
    epochs = range(1, len(training_history['train_losses']) + 1)
    base_filename = base_path.replace('.png', '')
    
    # Individual detailed plots
    
    # Plot 1: Training vs Validation Loss (detailed)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (Detailed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_filename}_detailed_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: GPA MSE only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['gpa_mse'], 'g-', label='GPA MSE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('GPA Mean Squared Error Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_filename}_gpa_mse_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: GPA MAE only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['gpa_mae'], 'purple', label='GPA MAE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('GPA Mean Absolute Error Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_filename}_gpa_mae_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: CPA MSE only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['cpa_mse'], 'orange', label='CPA MSE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('CPA Mean Squared Error Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_filename}_cpa_mse_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 5: CPA MAE only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_history['cpa_mae'], 'brown', label='CPA MAE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('CPA Mean Absolute Error Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_filename}_cpa_mae_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 6: Final Test Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = ['GPA MSE', 'GPA MAE', 'CPA MSE', 'CPA MAE']
    values = [
        test_metrics['gpa_mse'], 
        test_metrics['gpa_mae'], 
        test_metrics['cpa_mse'], 
        test_metrics['cpa_mae']
    ]
    colors = ['red', 'blue', 'orange', 'purple']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Error Value')
    plt.title('Final Test Metrics Comparison')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_filename}_final_metrics_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Detailed training metrics saved as separate files with prefix: {base_filename}_")
    return True

def main():
    parser = argparse.ArgumentParser(description='Student Performance Prediction Model')
    parser.add_argument('--course_csv', type=str, required=True, help='Path to course data CSV file')
    parser.add_argument('--performance_csv', type=str, required=True, help='Path to performance data CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_model', type=str, default='best_model.pth', help='Path to save the trained model')
    parser.add_argument('--output_plot', type=str, default='training_results.png', help='Path to save the training plot')
    parser.add_argument('--skip_dl', action='store_true', help='Skip deep learning training and only run traditional ML')
    parser.add_argument('--skip_ml', action='store_true', help='Skip traditional ML training and only run deep learning')
    
    args = parser.parse_args()
    
    print("=== MÔ HÌNH DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN ===")
    print("Architecture: Course Encoder + Transformer + LSTM + Attention")
    
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA không khả dụng! Kiểm tra:")
        print("1. Đã cài đặt NVIDIA GPU chưa?")
        print("2. Đã cài đặt NVIDIA Driver chưa?")
        print("3. Đã cài đặt CUDA Toolkit chưa?")
        print("4. Đã cài đặt PyTorch với CUDA support chưa?")
        print("\nChạy lệnh sau để cài đặt PyTorch với CUDA:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        device = torch.device('cpu')
    else:
        print("\nCUDA khả dụng!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Số GPU: {torch.cuda.device_count()}")
        print(f"GPU hiện tại: {torch.cuda.current_device()}")
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    
    print(f"\nSử dụng device: {device}")
    
    print("\nĐang tải dữ liệu...")
    print(f"Course CSV: {args.course_csv}")
    print(f"Performance CSV: {args.performance_csv}")
    
    course_df = pd.read_csv(args.course_csv)
    perf_df = pd.read_csv(args.performance_csv)
    
    print(f"Course data: {len(course_df)} records")
    print(f"Performance data: {len(perf_df)} records")
    
    print("\nĐang xử lý dữ liệu...")
    processor = StudentPerformanceDataProcessor()
    sequences = create_student_sequences(course_df, perf_df, processor)
    
    print(f"Đã tạo {len(sequences)} training sequences từ dữ liệu")
    
    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    train_sequences, val_sequences = train_test_split(train_sequences, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    train_dataset = StudentSequenceDataset(train_sequences)
    val_dataset = StudentSequenceDataset(val_sequences)
    test_dataset = StudentSequenceDataset(test_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)
    
    print("\nĐang khởi tạo mô hình...")
    model = StudentPerformancePredictor()
    model = model.to(device)
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("\nKiểm tra model trên GPU:")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    dl_results = None
    
    if not args.skip_dl:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        trainer = ModelTrainer(model, processor, device)
        
        print("\nBắt đầu training Deep Learning model...")
        training_history = trainer.train(train_loader, val_loader, epochs=args.epochs)
        
        print("\nĐang đánh giá mô hình trên test set...")
        trainer.model.load_state_dict(torch.load(args.output_model))
        dl_results = trainer.evaluate(test_loader)
    else:
        print("\nSkipping Deep Learning training...")
        training_history = None
        dl_results = None
    
    ml_results = None
    if not args.skip_ml:
        ml_trainer = TraditionalMLTrainer(processor)
        ml_results = ml_trainer.train_and_evaluate_all(train_sequences, test_sequences)
        ml_trainer.save_best_models(ml_results)
    else:
        print("\nSkipping Traditional ML training...")
    
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*50)
    
    if dl_results:
        print(f"DEEP LEARNING MODEL:")
        print(f"Test Loss: {dl_results['loss']:.4f}")
        print(f"\nGPA Prediction (Normalized):")
        print(f"  - MSE: {dl_results['gpa_mse']:.4f}")
        print(f"  - MAE: {dl_results['gpa_mae']:.4f}")
        print(f"  - R²: {dl_results['gpa_r2']:.4f}")
        print(f"\nCPA Prediction (Normalized):")
        print(f"  - MSE: {dl_results['cpa_mse']:.4f}")
        print(f"  - MAE: {dl_results['cpa_mae']:.4f}")
        print(f"  - R²: {dl_results['cpa_r2']:.4f}")
        
        if 'gpa_mse_denorm' in dl_results:
            print(f"\n=== KẾT QUẢ TRÊN DỮ LIỆU GỐC (KHÔNG NORMALIZE) ===")
            print(f"GPA Prediction (Denormalized):")
            print(f"  - MSE: {dl_results['gpa_mse_denorm']:.4f}")
            print(f"  - MAE: {dl_results['gpa_mae_denorm']:.4f}")
            print(f"  - R²: {dl_results['gpa_r2_denorm']:.4f}")
            print(f"\nCPA Prediction (Denormalized):")
            print(f"  - MSE: {dl_results['cpa_mse_denorm']:.4f}")
            print(f"  - MAE: {dl_results['cpa_mae_denorm']:.4f}")
            print(f"  - R²: {dl_results['cpa_r2_denorm']:.4f}")
    
    if ml_results:
        print(f"\n" + "="*60)
        print("BẢNG SO SÁNH CÁC MÔ HÌNH ML (DENORMALIZED METRICS)")
        print("="*60)
        
        print(f"{'Model':<20} {'GPA MAE':<10} {'GPA MSE':<10} {'CPA MAE':<10} {'CPA MSE':<10} {'Time(s)':<10}")
        print("-" * 70)
        
        for model_name, metrics in ml_results.items():
            if metrics is not None and 'gpa_mae_denorm' in metrics:
                print(f"{model_name:<20} {metrics['gpa_mae_denorm']:<10.4f} {metrics['gpa_mse_denorm']:<10.4f} "
                      f"{metrics['cpa_mae_denorm']:<10.4f} {metrics['cpa_mse_denorm']:<10.4f} {metrics['training_time']:<10.2f}")
        
        if dl_results and 'gpa_mae_denorm' in dl_results:
            print(f"{'Deep Learning':<20} {dl_results['gpa_mae_denorm']:<10.4f} {dl_results['gpa_mse_denorm']:<10.4f} "
                  f"{dl_results['cpa_mae_denorm']:<10.4f} {dl_results['cpa_mse_denorm']:<10.4f} {'N/A':<10}")
    
    if ml_results:
        plot_ml_comparison(ml_results, dl_results)
    
    base_filename = args.output_plot.replace('.png', '')
    
    if training_history:
        # Plot 1: Overall Training Loss
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['train_losses'], label='Train Loss', color='blue', linewidth=2)
        plt.plot(training_history['val_losses'], label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Overall Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_filename}_training_loss.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: MSE Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['gpa_mse'], label='GPA MSE', color='green', linewidth=2)
        plt.plot(training_history['cpa_mse'], label='CPA MSE', color='orange', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_filename}_mse_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: MAE Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['gpa_mae'], label='GPA MAE', color='purple', linewidth=2)
        plt.plot(training_history['cpa_mae'], label='CPA MAE', color='brown', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_filename}_mae_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: GPA Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['gpa_mse'], label='MSE', color='red', linewidth=2)
        plt.plot(training_history['gpa_mae'], label='MAE', color='blue', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('GPA Prediction Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_filename}_gpa_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 5: CPA Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(training_history['cpa_mse'], label='MSE', color='red', linewidth=2)
        plt.plot(training_history['cpa_mae'], label='MAE', color='blue', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('CPA Prediction Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_filename}_cpa_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if dl_results:
            with torch.no_grad():
                trainer.model.eval()
                gpa_predictions = []
                gpa_targets = []
                cpa_predictions = []
                cpa_targets = []
                
                for batch in test_loader:
                    course_features = batch['course_features'].to(device)
                    semester_features = batch['semester_features'].to(device)
                    course_masks = batch['course_masks'].to(device)
                    semester_masks = batch['semester_masks'].to(device)
                    target_gpa = batch['target_gpa']
                    target_cpa = batch['target_cpa']
                    
                    predictions = trainer.model(course_features, semester_features, course_masks, semester_masks)
                    
                    gpa_predictions.extend(predictions[:, 0].cpu().numpy())
                    gpa_targets.extend(target_gpa.numpy())
                    cpa_predictions.extend(predictions[:, 1].cpu().numpy())
                    cpa_targets.extend(target_cpa.numpy())
            
            # Plot 6: GPA Prediction Scatter
            plt.figure(figsize=(10, 6))
            gpa_min = min(min(gpa_targets), min(gpa_predictions))
            gpa_max = max(max(gpa_targets), max(gpa_predictions))
            plt.scatter(gpa_targets, gpa_predictions, alpha=0.6, color='blue', s=30)
            plt.plot([gpa_min, gpa_max], [gpa_min, gpa_max], 'r--', alpha=0.8, label='Perfect Prediction')
            plt.xlabel('Actual GPA')
            plt.ylabel('Predicted GPA')
            plt.title(f'GPA Prediction Scatter Plot (R²={dl_results["gpa_r2"]:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{base_filename}_gpa_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 7: CPA Prediction Scatter
            plt.figure(figsize=(10, 6))
            cpa_min = min(min(cpa_targets), min(cpa_predictions))
            cpa_max = max(max(cpa_targets), max(cpa_predictions))
            plt.scatter(cpa_targets, cpa_predictions, alpha=0.6, color='green', s=30)
            plt.plot([cpa_min, cpa_max], [cpa_min, cpa_max], 'r--', alpha=0.8, label='Perfect Prediction')
            plt.xlabel('Actual CPA')
            plt.ylabel('Predicted CPA')
            plt.title(f'CPA Prediction Scatter Plot (R²={dl_results["cpa_r2"]:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{base_filename}_cpa_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 8: Combined Prediction Comparison
            plt.figure(figsize=(10, 6))
            min_val = min(min(gpa_targets), min(gpa_predictions), min(cpa_targets), min(cpa_predictions))
            max_val = max(max(gpa_targets), max(gpa_predictions), max(cpa_targets), max(cpa_predictions))
            plt.scatter(gpa_targets, gpa_predictions, alpha=0.6, color='blue', label=f'GPA (R²={dl_results["gpa_r2"]:.3f})', s=30)
            plt.scatter(cpa_targets, cpa_predictions, alpha=0.6, color='green', label=f'CPA (R²={dl_results["cpa_r2"]:.3f})', s=30)
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('GPA vs CPA Prediction Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{base_filename}_combined_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            detailed_plot_path = args.output_plot.replace('.png', '_detailed_metrics')
            plot_detailed_training_metrics(training_history, dl_results, detailed_plot_path)
    
    print(f"\nSummary of Training Progress:")
    if training_history:
        print(f"Total epochs: {len(training_history['train_losses'])}")
        print(f"Best GPA MSE: {min(training_history['gpa_mse']):.4f} at epoch {training_history['gpa_mse'].index(min(training_history['gpa_mse'])) + 1}")
        print(f"Best GPA MAE: {min(training_history['gpa_mae']):.4f} at epoch {training_history['gpa_mae'].index(min(training_history['gpa_mae'])) + 1}")
        print(f"Best CPA MSE: {min(training_history['cpa_mse']):.4f} at epoch {training_history['cpa_mse'].index(min(training_history['cpa_mse'])) + 1}")
        print(f"Best CPA MAE: {min(training_history['cpa_mae']):.4f} at epoch {training_history['cpa_mae'].index(min(training_history['cpa_mae'])) + 1}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("\nThông tin GPU sau khi training:")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(processor.performance_scaler, f)
    print(f"Đã lưu performance scaler vào 'feature_scaler.pkl'")
    
    print(f"\nĐã lưu tất cả plots riêng biệt với prefix:")
    print(f"  - Main plots: '{base_filename}_*.png'")
    if training_history:
        print(f"  - Detailed plots: '{args.output_plot.replace('.png', '_detailed_metrics')}_*.png'")
    print(f"  - Model weights: '{args.output_model}'")
    print(f"  - Performance scaler: 'feature_scaler.pkl'")
    if ml_results:
        print(f"  - Traditional ML comparison: 'ml_algorithms_comparison.png'")
    
    if training_history:
        print(f"\nDanh sách file plots đã tạo:")
        plot_files = [
            f"{base_filename}_training_loss.png",
            f"{base_filename}_mse_comparison.png", 
            f"{base_filename}_mae_comparison.png",
            f"{base_filename}_gpa_metrics.png",
            f"{base_filename}_cpa_metrics.png",
        ]
        
        if dl_results:
            plot_files.extend([
                f"{base_filename}_gpa_scatter.png",
                f"{base_filename}_cpa_scatter.png",
                f"{base_filename}_combined_scatter.png",
            ])
            
            detailed_plot_path = args.output_plot.replace('.png', '_detailed_metrics')
            plot_files.extend([
                f"{detailed_plot_path}_detailed_loss.png",
                f"{detailed_plot_path}_gpa_mse_only.png",
                f"{detailed_plot_path}_gpa_mae_only.png",
                f"{detailed_plot_path}_cpa_mse_only.png",
                f"{detailed_plot_path}_cpa_mae_only.png",
                f"{detailed_plot_path}_final_metrics_bar.png"
            ])
        
        for i, file in enumerate(plot_files, 1):
            print(f"  {i:2d}. {file}")
    
    if dl_results and 'gpa_mse_denorm' in dl_results:
        print(f"\n=== BẢNG SO SÁNH METRICS (DEEP LEARNING) ===")
        print(f"{'Metric':<20} {'Normalized':<15} {'Denormalized':<15}")
        print("-" * 50)
        print(f"{'GPA MSE':<20} {dl_results['gpa_mse']:<15.4f} {dl_results['gpa_mse_denorm']:<15.4f}")
        print(f"{'GPA MAE':<20} {dl_results['gpa_mae']:<15.4f} {dl_results['gpa_mae_denorm']:<15.4f}")
        print(f"{'CPA MSE':<20} {dl_results['cpa_mse']:<15.4f} {dl_results['cpa_mse_denorm']:<15.4f}")
        print(f"{'CPA MAE':<20} {dl_results['cpa_mae']:<15.4f} {dl_results['cpa_mae_denorm']:<15.4f}")

if __name__ == "__main__":
    main() 